"""
Tests for app.py's document-attachment, voice-input, and Phase 6 approval UI
wiring.

These drive the real Streamlit script headlessly via streamlit.testing.v1's
AppTest (no browser, no real Gemini/network calls - agent.run_agent_turn and
agent.transcribe_audio are monkeypatched). The default page is Chat (the
primary page); several tests switch st.session_state["nav_view"] to reach
the Documents/Changes pages where attachment clearing and change approval
now live in the redesigned UI.
"""

import re
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import workflow

_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _clean_workflow_registry():
    """Phase 6's pending-change registry and repair-attempt tallies are
    process-wide module-level stores (see workflow.py) - reset them around
    every test in this file."""
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()
    yield
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()


@pytest.fixture
def apptest_with_mocked_agent(monkeypatch):
    """Run app.py headlessly with run_agent_turn/transcribe_audio replaced by
    fakes, so no real API key or network call is ever needed."""
    import agent as agent_module

    def fake_run_agent_turn(agent, history):
        last = history[-1]
        text = getattr(last, "content", str(last))
        return {"answer": f"MOCKED REPLY for: {text[:200]}", "tool_calls": []}

    def fake_transcribe_audio(audio_bytes, mime_type="audio/wav"):
        return "explain how tools.py works in my project"

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr(agent_module, "transcribe_audio", fake_transcribe_audio)

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert at.exception == []
    return at


def _fake_uploaded_file(name: str, content: bytes, mime_type: str = "text/plain"):
    """A minimal stand-in for streamlit's UploadedFile: just needs .name and
    .getvalue() for documents.process_upload()."""
    return types.SimpleNamespace(name=name, type=mime_type, getvalue=lambda: content)


def _fake_chat_value(text="", files=None, audio=None):
    return types.SimpleNamespace(text=text, files=files or [], audio=audio)


def _goto(at, page: str):
    """Switch the redesigned app's page router and re-run."""
    at.session_state["nav_view"] = page
    at.run(timeout=30)
    assert at.exception == []
    return at


def test_no_files_attached_initially(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    assert at.session_state["attached_files"] == []


def test_attaching_a_supported_file_updates_session_state(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    from documents import process_upload

    processed = process_upload("example.py", b"def add(a, b):\n    return a + b\n")
    at.session_state["attached_files"] = [processed]
    at.run(timeout=30)
    assert at.exception == []

    attached = at.session_state["attached_files"]
    assert len(attached) == 1
    assert attached[0]["filename"] == "example.py"


def test_clear_attachments_button_empties_attached_files(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    from documents import process_upload

    at.session_state["attached_files"] = [
        process_upload("example.py", b"print('hi')\n")
    ]
    _goto(at, "Documents")
    assert at.session_state["attached_files"] != []

    clear_btn = next(b for b in at.button if b.label == "Clear all attachments")
    clear_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["attached_files"] == []


def test_clearing_attachments_does_not_touch_conversation(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    from documents import process_upload

    # Send one normal message first (Chat is the default page).
    at.chat_input[0].set_value("What is Python?").run(timeout=30)
    assert at.exception == []
    message_count_before = len(at.session_state["messages"])
    assert message_count_before > 0

    at.session_state["attached_files"] = [process_upload("a.py", b"x = 1\n")]
    _goto(at, "Documents")
    clear_btn = next(b for b in at.button if b.label == "Clear all attachments")
    clear_btn.click()
    at.run(timeout=30)

    assert at.session_state["attached_files"] == []
    assert len(at.session_state["messages"]) == message_count_before


def test_attached_document_content_reaches_the_agent_as_context(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    from documents import process_upload

    at.session_state["attached_files"] = [
        process_upload("example.py", b"def mystery_function():\n    return 42\n")
    ]
    at.run(timeout=30)
    assert at.exception == []

    at.chat_input[0].set_value("Explain this file.").run(timeout=30)
    assert at.exception == []

    sent_history = at.session_state["lc_history"]
    last_human_message = sent_history[-2]
    assert "mystery_function" in last_human_message.content
    assert "untrusted" in last_human_message.content.lower()

    # The chat bubble itself must show only what the user actually typed,
    # never the injected document-context block.
    displayed_messages = at.session_state["messages"]
    assert displayed_messages[-2]["content"] == "Explain this file."


def test_voice_input_is_transcribed_and_flows_through_normal_chat(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import app as app_module

    monkeypatch.setattr(
        app_module.st,
        "chat_input",
        lambda *a, **kw: _fake_chat_value(
            text="", audio=_fake_uploaded_file("voice.wav", b"fake-audio", "audio/wav")
        ),
    )
    at.run(timeout=30)
    assert at.exception == []

    messages = at.session_state["messages"]
    assert messages[-2]["content"] == "explain how tools.py works in my project"
    assert messages[-2]["source"] == "voice"
    assert messages[-1]["content"].startswith("MOCKED REPLY")


def test_attaching_file_with_question_in_one_submission_updates_state_same_turn(
    apptest_with_mocked_agent, monkeypatch
):
    # Regression test: attachment handling and the chat-input handling code
    # live in the same render pass - without an internal rerun, a newly
    # attached file would only be reflected in session_state on the *next*
    # interaction, not the one that attached it.
    at = apptest_with_mocked_agent

    import app as app_module

    # Like the real widget, the fake chat_input must only "submit" once -
    # every rerun after that (including the extra internal st.rerun() this
    # fix performs) must see an empty input, or the app would loop forever.
    call_count = {"n": 0}

    def fake_chat_input(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _fake_chat_value(
                text="Explain this file.",
                files=[
                    _fake_uploaded_file(
                        "example.py", b"def add(a, b):\n    return a + b\n"
                    )
                ],
            )
        return None

    monkeypatch.setattr(app_module.st, "chat_input", fake_chat_input)
    at.run(timeout=30)
    assert at.exception == []

    attached = at.session_state["attached_files"]
    assert len(attached) == 1
    assert attached[0]["filename"] == "example.py"

    messages = at.session_state["messages"]
    assert messages[-2]["content"] == "Explain this file."
    assert messages[-1]["content"].startswith("MOCKED REPLY")


def test_unsupported_file_upload_is_rejected_and_not_attached(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import app as app_module

    monkeypatch.setattr(
        app_module.st,
        "chat_input",
        lambda *a, **kw: _fake_chat_value(
            text="", files=[_fake_uploaded_file("malicious.exe", b"MZ...")]
        ),
    )
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["attached_files"] == []


def test_oversized_file_upload_is_rejected_and_not_attached(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import app as app_module
    import documents

    monkeypatch.setattr(
        app_module.st,
        "chat_input",
        lambda *a, **kw: _fake_chat_value(
            text="",
            files=[
                _fake_uploaded_file(
                    "big.txt", b"a" * (documents.MAX_UPLOAD_SIZE_BYTES + 1)
                )
            ],
        ),
    )
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["attached_files"] == []


# ---------------------------------------------------------------------------
# Regression tests: clicking a nav-switching control used to crash with
# `StreamlitAPIException: st.session_state.nav_view cannot be modified
# after the widget with key nav_view is instantiated` - found via live
# manual testing, not by the suite (none of these clicked before). The fix
# (_request_nav_change) hands the target page off through a plain
# `_nav_request` key that the sidebar consumes *before* re-creating the
# nav_view-bound radio widget, instead of writing nav_view directly from a
# button handler that runs after that widget.
# ---------------------------------------------------------------------------


def test_header_settings_button_switches_page_without_crashing(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    settings_btn = next(b for b in at.button if b.key == "header_settings")
    settings_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["nav_view"] == "Settings"


def test_sidebar_review_changes_button_switches_page_without_crashing(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)
    review_btn = next(b for b in at.sidebar.button if b.label == "Review changes")
    review_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["nav_view"] == "Changes"


def test_home_shortcut_switches_to_chat_and_submits_without_crashing(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    at.session_state["nav_view"] = "Home"
    at.run(timeout=30)
    home_btn = next(b for b in at.button if b.key == "home_Run Tests")
    home_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["nav_view"] == "Chat"
    assert at.session_state["messages"][-1]["content"].startswith("MOCKED REPLY")


def test_injected_style_block_has_no_blank_lines():
    """Regression test for a real bug found via visual (Playwright)
    inspection: Streamlit's markdown-to-HTML pass treats a raw HTML block
    as ended by the first blank line inside it (a CommonMark HTML-block
    rule), so a blank line inside the app's injected <style>...</style>
    silently truncated the stylesheet mid-parse - every CSS rule after
    that point rendered as literal visible page text instead of being
    applied as styling. No AppTest-based test (which only checks for
    Python exceptions/session state, never rendered visible text) could
    have caught this - only an actual screenshot did."""
    app_source = Path(__file__).resolve().parent.parent / "app.py"
    text = app_source.read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", text, re.DOTALL)
    assert match is not None, "app.py must inject a <style> block"
    style_body = match.group(1)
    lines = style_body.split("\n")
    # The very last split segment is just the whitespace indentation before
    # the closing </style> tag on its own line, e.g. "    </style>" - that
    # trailing indentation is not a real blank *line* in the source file
    # (nothing follows it inside the block), so it's excluded here.
    blank_line_numbers = [
        index for index, line in enumerate(lines[:-1]) if line.strip() == ""
    ]
    assert blank_line_numbers == [], (
        f"Found blank line(s) inside <style> at offset(s) {blank_line_numbers} "
        "within the block - this truncates the stylesheet early and leaks "
        "the remaining CSS as visible page text."
    )


def test_capabilities_panel_lists_all_five_phases_with_no_buttons(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent

    phase_labels = [e.label for e in at.sidebar.expander]
    assert any("Phase 1" in label for label in phase_labels)
    assert any("Phase 2" in label for label in phase_labels)
    assert any("Phase 3" in label for label in phase_labels)
    assert any("Phase 4" in label for label in phase_labels)
    assert any("Phase 5" in label for label in phase_labels)

    phase1 = next(e for e in at.sidebar.expander if "Phase 1" in e.label)
    # Information only: no buttons inside a phase's reference panel.
    assert len(phase1.button) == 0
    phase1_text = "\n".join(m.value for m in phase1.markdown)
    assert "Calculator" in phase1_text
    assert "Example capability" in phase1_text


def test_expanding_capabilities_panel_does_not_touch_chat_or_pending_prompt(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent

    messages_before = list(at.session_state["messages"])
    assert at.session_state["pending_prompt"] is None

    # Merely having the Capabilities expanders rendered (as they are on
    # every run of this fixture) must never insert, select, or submit
    # a question.
    at.run(timeout=30)
    assert at.exception == []

    assert at.session_state["messages"] == messages_before
    assert at.session_state["pending_prompt"] is None


# ---------------------------------------------------------------------------
# Phase 6: the Changes page (propose -> human approve/reject -> apply).
# Approving now always drives the real apply-and-test continuation - the
# old "sidebar approve is mark-only, chat approve resumes" split has been
# deliberately retired in favor of one consolidated Approve action.
# ---------------------------------------------------------------------------


def test_no_pending_changes_shows_empty_state(apptest_with_mocked_agent):
    at = _goto(apptest_with_mocked_agent, "Changes")
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "No pending changes" in markdown_text


def test_pending_change_appears_with_approve_reject_buttons(
    apptest_with_mocked_agent,
):
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(apptest_with_mocked_agent, "Changes")

    markdown_text = "\n".join(m.value for m in at.markdown)
    caption_text = "\n".join(c.value for c in at.caption)
    assert change.file_path in markdown_text
    assert change.change_id in caption_text

    button_keys = {b.key for b in at.button}
    assert f"approve_{change.change_id}" in button_keys
    assert f"reject_{change.change_id}" in button_keys


def test_clicking_approve_marks_the_change_approved_and_resumes(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(at, "Changes")

    approve_btn = next(b for b in at.button if b.key == f"approve_{change.change_id}")
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id).approved is True
    # Approving now resumes the real turn-runner (mocked in this fixture),
    # so a new assistant reply should have been recorded too.
    assert at.session_state["messages"][-1]["content"].startswith("MOCKED REPLY")


def test_clicking_reject_removes_pending_change(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(at, "Changes")

    reject_btn = next(b for b in at.button if b.key == f"reject_{change.change_id}")
    reject_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id) is None
    assert workflow.list_pending_changes() == []


def test_workflow_states_are_displayed_when_present(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import agent as agent_module

    def fake_run_agent_turn_with_workflow(agent, history):
        return {
            "answer": "Done.",
            "tool_calls": [],
            "workflow_states": ["IMPLEMENTING", "WAITING_FOR_APPROVAL"],
        }

    monkeypatch.setattr(
        agent_module, "run_agent_turn", fake_run_agent_turn_with_workflow
    )
    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    assert at.exception == []

    last_message = at.session_state["messages"][-1]
    assert last_message["workflow_states"] == ["IMPLEMENTING", "WAITING_FOR_APPROVAL"]


# ---------------------------------------------------------------------------
# Phase 6: repair-attempt circuit breaker reset control (now on Changes page)
# ---------------------------------------------------------------------------


def test_no_repair_warning_when_no_file_has_hit_the_limit(apptest_with_mocked_agent):
    at = _goto(apptest_with_mocked_agent, "Changes")
    warning_text = "\n".join(w.value for w in at.warning)
    assert "repair-attempt limit" not in warning_text.lower()
    reset_buttons = [b for b in at.button if "Reset repair counter" in b.label]
    assert reset_buttons == []


def test_repair_warning_and_reset_button_appear_once_limit_reached(
    apptest_with_mocked_agent,
):
    for _ in range(workflow.MAX_REPAIR_ATTEMPTS):
        workflow.record_apply("stuck_file.py")
    at = _goto(apptest_with_mocked_agent, "Changes")

    warning_text = "\n".join(w.value for w in at.warning)
    assert "repair-attempt limit" in warning_text.lower()
    assert "stuck_file.py" in warning_text

    reset_buttons = [b for b in at.button if "Reset repair counter" in b.label]
    assert len(reset_buttons) == 1


def test_clicking_reset_repair_counter_clears_the_tally(apptest_with_mocked_agent):
    for _ in range(workflow.MAX_REPAIR_ATTEMPTS):
        workflow.record_apply("stuck_file.py")
    at = _goto(apptest_with_mocked_agent, "Changes")

    reset_btn = next(b for b in at.button if "Reset repair counter" in b.label)
    reset_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.repair_attempts_for("stuck_file.py") == 0


def test_approve_button_resumes_the_workflow_with_full_details(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import agent as agent_module

    calls = []

    def fake_run_agent_turn_resuming(agent, history):
        calls.append(history[-1].content)
        return {
            "answer": "Applied the change and all tests pass.",
            "tool_calls": [
                {
                    "name": "apply_approved_change",
                    "input": {"change_id": "abc123"},
                    "output": "ok",
                }
            ],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_resuming)

    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    workflow._pending_changes.pop(change.change_id)
    change.change_id = "abc123"
    workflow._pending_changes["abc123"] = change

    at = _goto(at, "Changes")

    approve_btn = next(b for b in at.button if b.key == "approve_abc123")
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change("abc123").approved is True
    assert calls, "run_agent_turn should have been called to resume the workflow"
    assert "abc123" in calls[-1]

    messages = at.session_state["messages"]
    assert messages[-1]["content"] == "Applied the change and all tests pass."
    assert messages[-1]["workflow_states"][-1] == "COMPLETED"
    assert messages[-2]["content"].startswith("✅ Approved change")
