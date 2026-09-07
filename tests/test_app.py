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

import io
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
    fakes, so no real API key or network call is ever needed.

    GOOGLE_API_KEY is explicitly set to a format-valid fake value: app.py's
    own module-level code still calls the REAL build_agent()/_build_llm()
    (only run_agent_turn/transcribe_audio are mocked above), so this suite
    must never depend on whatever happens to be in the real local .env file
    - it would otherwise pass or fail based on unrelated machine state.
    """
    import agent as agent_module

    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyD-fake1234567890abcdefghijklmno")

    def fake_run_agent_turn(agent, history, **kwargs):
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


# ---------------------------------------------------------------------------
# API key configuration (agent.api_key_looks_valid) - a genuinely missing
# key must still show a clear error, but a configured key that merely
# doesn't match the traditional "AIza..." shape must NOT be blocked (only a
# real rejection from Google's API counts). Either way, the actual
# configured key value must never appear anywhere in the rendered UI.
# ---------------------------------------------------------------------------

_FAKE_INVALID_FORMAT_KEY = "AQ.FakeNonGeminiTokenForTestingOnly1234567890"


def _rendered_ui_text(at) -> str:
    """Every bit of text this run actually rendered - markdown, errors,
    captions, info notices - concatenated, to search for an accidental
    secret leak (or, for other tests, confirm a specific notice is/isn't
    shown)."""
    return "\n".join(
        [m.value for m in at.markdown]
        + [e.value for e in at.error]
        + [c.value for c in at.caption]
        + [i.value for i in at.info]
    )


def test_missing_api_key_shows_clear_error_and_stops(monkeypatch):
    # Set to "" rather than delenv(): AppTest re-runs agent.py's own
    # module-level load_dotenv() in its script execution context, which
    # (by design - override=False) reloads GOOGLE_API_KEY from the real
    # local .env file the moment the variable is fully absent from
    # os.environ. Setting it to an empty string keeps the variable
    # "present" (just empty), which load_dotenv() correctly leaves alone -
    # this is what actually simulates "missing/empty" without the real
    # local .env's value silently reappearing mid-test.
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert at.exception == []
    assert "not configured" in _rendered_ui_text(at).lower()


def test_differently_shaped_api_key_does_not_block_the_app(monkeypatch):
    """A Google credential that doesn't match the traditional 'AIza...'
    shape (e.g. a working 'AQ.'-prefixed key, seen in a real support case)
    must not be hard-rejected locally - the app must load normally instead
    of stopping on the old format error, leaving it to Google's own API to
    determine whether the credential actually works. It also must not show
    ANY format-mismatch message in the UI (regression test: the app used to
    show an "doesn't match the traditional 'AIza...' Gemini key format"
    st.info() notice here - api_key_looks_valid is advisory-only, so the UI
    must stay silent about format and let a real API rejection, not a local
    guess, be the only thing that ever tells the user their key is wrong)."""
    monkeypatch.setenv("GOOGLE_API_KEY", _FAKE_INVALID_FORMAT_KEY)
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    assert at.exception == []
    rendered = _rendered_ui_text(at)
    assert "does not look like a valid gemini api key" not in rendered.lower()
    assert "doesn't match the traditional" not in rendered.lower()
    assert "aiza" not in rendered.lower()
    # The actual (fake, but still "the configured value") key must never
    # appear anywhere in the rendered page.
    assert _FAKE_INVALID_FORMAT_KEY not in rendered


def test_valid_api_key_never_appears_anywhere_in_the_rendered_ui(
    apptest_with_mocked_agent,
):
    """apptest_with_mocked_agent sets a real-shaped fake GOOGLE_API_KEY (see
    its own docstring) - confirm that value is never echoed anywhere in the
    app's rendered output, including the Settings page's env-var status
    panel (which must show presence only, never the value)."""
    at = _goto(apptest_with_mocked_agent, "Settings")
    rendered = _rendered_ui_text(at)
    assert "AIzaSyD-fake1234567890abcdefghijklmno" not in rendered
    assert "configured" in rendered.lower()


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
# Phase 6: the Changes page (propose -> human approve -> apply). Approval
# and apply are now change-SET-level actions: exactly one "Approve Changes"
# button and, only once approved, exactly one "Apply Approved Changes"
# button for the WHOLE proposed set - never one button per file. The two
# are deliberately separate: Approve only moves state PENDING -> APPROVED
# (no file written, no test run, no agent call); Apply is the one action
# that writes the approved files and resumes the agent for testing/fixing.
# ---------------------------------------------------------------------------


@pytest.fixture
def _apply_scratch_files():
    """Disposable project-relative paths for tests that click the real
    Apply button - tools.apply_approved_change_set() writes real files to
    disk (run_agent_turn stays mocked, but the deterministic apply step
    itself is real, by design), so anything registered here is removed
    after the test regardless of outcome."""
    created: list[Path] = []

    def _register(rel_path: str) -> str:
        created.append(Path(__file__).resolve().parent.parent / rel_path)
        return rel_path

    yield _register
    for path in created:
        if path.exists():
            path.unlink()


def _messages_starting_with(at, prefix: str) -> list[dict]:
    return [m for m in at.session_state["messages"] if m["content"].startswith(prefix)]


def test_no_pending_changes_shows_empty_state(apptest_with_mocked_agent):
    at = _goto(apptest_with_mocked_agent, "Changes")
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "No pending changes" in markdown_text


def test_one_proposed_change_shows_a_single_approve_button_no_apply_yet(
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
    assert "TOTAL: 1 change(s)" in caption_text

    button_keys = {b.key for b in at.button}
    assert f"approve_set_{change.changeset_id}" in button_keys
    assert f"reject_set_{change.changeset_id}" in button_keys
    # Not approved yet - Apply must not be offered.
    assert f"apply_set_{change.changeset_id}" not in button_keys
    assert len([k for k in button_keys if k and k.startswith("approve_")]) == 1


def test_multiple_proposed_changes_in_one_task_share_a_single_changeset(
    apptest_with_mocked_agent,
):
    """Two files proposed for the same development task (e.g. a source file
    and its test file) must be reviewed as ONE change set - a single
    Approve button covering both, never one button per file."""
    change_a = workflow.register_change(
        file_path="a.py", action="create", content="x = 1\n", reason="demo a"
    )
    change_b = workflow.register_change(
        file_path="b.py", action="create", content="y = 2\n", reason="demo b"
    )
    assert change_a.changeset_id == change_b.changeset_id

    at = _goto(apptest_with_mocked_agent, "Changes")
    markdown_text = "\n".join(m.value for m in at.markdown)
    caption_text = "\n".join(c.value for c in at.caption)
    assert change_a.file_path in markdown_text
    assert change_b.file_path in markdown_text
    assert "TOTAL: 2 change(s)" in caption_text

    approve_buttons = [
        b for b in at.button if b.key and b.key.startswith("approve_set_")
    ]
    assert len(approve_buttons) == 1


def test_approving_the_change_set_approves_every_member_with_one_message(
    apptest_with_mocked_agent,
):
    change_a = workflow.register_change(
        file_path="a.py", action="create", content="x = 1\n", reason="demo a"
    )
    change_b = workflow.register_change(
        file_path="b.py", action="create", content="y = 2\n", reason="demo b"
    )
    at = _goto(apptest_with_mocked_agent, "Changes")

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change_a.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change_a.change_id).approved is True
    assert workflow.get_change(change_b.change_id).approved is True
    # Approve never applies - no file is written, whatever the outcome.
    assert workflow.get_change(change_a.change_id).applied is False
    assert workflow.get_change(change_b.change_id).applied is False
    # Exactly one confirmation for the whole set, never one per file.
    approval_messages = _messages_starting_with(
        at, "✅ All 2 proposed changes have been approved"
    )
    assert len(approval_messages) == 1


def test_approving_a_single_change_set_member_uses_singular_wording(
    apptest_with_mocked_agent,
):
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(apptest_with_mocked_agent, "Changes")

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert _messages_starting_with(at, "✅ The proposed change has been approved.")


def test_approving_does_not_write_files_run_tests_or_call_the_agent(monkeypatch):
    """Approve moves state PENDING -> APPROVED only - it must never write a
    file, run tests, or call the agent (Apply is the separate action that
    does that)."""
    import agent as agent_module

    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyD-fake1234567890abcdefghijklmno")
    call_count = {"n": 0}

    def counting_run_agent_turn(agent, history, **kwargs):
        call_count["n"] += 1
        return {"answer": "MOCKED", "tool_calls": []}

    monkeypatch.setattr(agent_module, "run_agent_turn", counting_run_agent_turn)

    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = AppTest.from_file(_APP_PATH)
    at.session_state["nav_view"] = "Changes"
    at.run(timeout=30)
    assert at.exception == []

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert call_count["n"] == 0
    assert workflow.get_change(change.change_id).applied is False


def test_rerun_after_approval_does_not_reapprove_or_duplicate_the_message(
    apptest_with_mocked_agent,
):
    """Regression coverage for the reported bug: once a change set has been
    approved, a later Streamlit rerun (no new interaction - e.g. a reconnect
    or an unrelated re-render) must not emit a second approval message, must
    not show the Approve button again, and must not re-approve anything."""
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(at, "Changes")

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert (
        len(_messages_starting_with(at, "✅ The proposed change has been approved."))
        == 1
    )
    message_count_after_approval = len(at.session_state["messages"])

    # Still approved-but-unapplied - simulate a Streamlit rerun with no new
    # user interaction, e.g. a reconnect or a re-render triggered by
    # something unrelated on the page.
    at.run(timeout=30)
    assert at.exception == []

    assert (
        len(_messages_starting_with(at, "✅ The proposed change has been approved."))
        == 1
    )
    assert len(at.session_state["messages"]) == message_count_after_approval
    button_keys = {b.key for b in at.button}
    assert f"approve_set_{change.changeset_id}" not in button_keys
    # Apply is now the only offered action.
    assert f"apply_set_{change.changeset_id}" in button_keys


def test_clicking_reject_removes_the_whole_change_set(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    change_a = workflow.register_change(
        file_path="a.py", action="create", content="x = 1\n", reason="demo a"
    )
    change_b = workflow.register_change(
        file_path="b.py", action="create", content="y = 2\n", reason="demo b"
    )
    at = _goto(at, "Changes")

    reject_btn = next(
        b for b in at.button if b.key == f"reject_set_{change_a.changeset_id}"
    )
    reject_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change_a.change_id) is None
    assert workflow.get_change(change_b.change_id) is None
    assert workflow.list_pending_changes() == []


# ---------------------------------------------------------------------------
# Phase 6: a pending change set must be reviewable inline on the Chat page
# too - not only on the separate Changes page. Regression coverage for a
# reported bug where a user who stayed on Chat (the default page, right
# where the assistant proposes a change) saw no Approve control at all,
# only a passive "see the Changes page" banner.
# ---------------------------------------------------------------------------


def test_pending_changeset_appears_inline_on_chat_page(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)  # nav_view is "Chat" by default - no navigation needed
    assert at.exception == []

    markdown_text = "\n".join(m.value for m in at.markdown)
    caption_text = "\n".join(c.value for c in at.caption)
    assert change.file_path in markdown_text
    assert change.change_id in caption_text

    button_keys = {b.key for b in at.button}
    assert f"approve_set_{change.changeset_id}" in button_keys
    assert f"reject_set_{change.changeset_id}" in button_keys


def test_clicking_approve_on_chat_page_approves_without_calling_the_agent(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id).approved is True
    assert workflow.get_change(change.change_id).applied is False
    assert at.session_state["messages"][-1]["content"].startswith(
        "✅ The proposed change has been approved."
    )


def test_clicking_reject_on_chat_page_removes_pending_changeset(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)

    reject_btn = next(
        b for b in at.button if b.key == f"reject_set_{change.changeset_id}"
    )
    reject_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id) is None
    assert workflow.list_pending_changes() == []


def test_apply_button_only_appears_after_approval(apptest_with_mocked_agent):
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(apptest_with_mocked_agent, "Changes")
    assert f"apply_set_{change.changeset_id}" not in {b.key for b in at.button}

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    button_keys = {b.key for b in at.button}
    assert f"apply_set_{change.changeset_id}" in button_keys
    assert f"approve_set_{change.changeset_id}" not in button_keys
    assert len([k for k in button_keys if k and k.startswith("apply_set_")]) == 1


def test_applying_writes_every_approved_change_exactly_once(
    apptest_with_mocked_agent, _apply_scratch_files
):
    at = apptest_with_mocked_agent
    path_a = _apply_scratch_files("tests/_phase6_ui_scratch_a.py")
    path_b = _apply_scratch_files("tests/_phase6_ui_scratch_b.py")
    change_a = workflow.register_change(
        file_path=path_a, action="create", content="a = 1\n", reason="demo a"
    )
    change_b = workflow.register_change(
        file_path=path_b, action="create", content="b = 2\n", reason="demo b"
    )
    at = _goto(at, "Changes")

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change_a.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change_a.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change_a.change_id).applied is True
    assert workflow.get_change(change_b.change_id).applied is True
    assert (Path(__file__).resolve().parent.parent / path_a).read_text() == "a = 1\n"
    assert (Path(__file__).resolve().parent.parent / path_b).read_text() == "b = 2\n"
    assert _messages_starting_with(at, "✅ 2 approved changes applied successfully.")


def test_apply_button_disappears_once_change_set_fully_applied(
    apptest_with_mocked_agent, _apply_scratch_files
):
    at = apptest_with_mocked_agent
    path = _apply_scratch_files("tests/_phase6_ui_scratch_c.py")
    change = workflow.register_change(
        file_path=path, action="create", content="x = 1\n", reason="demo"
    )
    at = _goto(at, "Changes")

    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    # The click's own run still reflects the Apply button as it was drawn
    # BEFORE its handler ran this pass (Streamlit doesn't retroactively erase
    # already-drawn widgets mid-script) - a further rerun (the same thing any
    # later interaction or reconnect would trigger) is what actually reflects
    # the change set being gone now that every member is applied.
    at.run(timeout=30)
    assert at.exception == []

    button_keys = {b.key for b in at.button}
    assert f"approve_set_{change.changeset_id}" not in button_keys
    assert f"apply_set_{change.changeset_id}" not in button_keys


def test_workflow_states_are_displayed_when_present(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import agent as agent_module

    def fake_run_agent_turn_with_workflow(agent, history, **kwargs):
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


def test_apply_button_resumes_the_workflow_with_full_details(
    apptest_with_mocked_agent, monkeypatch, _apply_scratch_files
):
    at = apptest_with_mocked_agent
    path = _apply_scratch_files("tests/_phase6_ui_scratch_resume.py")

    import agent as agent_module

    calls = []

    def fake_run_agent_turn_resuming(agent, history, **kwargs):
        calls.append(history[-1].content)
        return {
            "answer": "Applied the change and all tests pass.",
            "tool_calls": [],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_resuming)

    change = workflow.register_change(
        file_path=path, action="create", content="x = 1\n", reason="demo"
    )
    workflow._pending_changes.pop(change.change_id)
    change.change_id = "abc123"
    workflow._pending_changes["abc123"] = change

    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change("abc123").applied is True
    assert calls, "run_agent_turn should have been called to resume the workflow"
    assert "abc123" in calls[-1]

    messages = at.session_state["messages"]
    assert messages[-1]["content"] == "Applied the change and all tests pass."
    assert messages[-1]["workflow_states"][-1] == "COMPLETED"
    assert messages[-2]["content"].startswith(
        "✅ 1 approved change applied successfully."
    )


def test_applied_change_disappears_from_pending_and_cannot_be_reapproved(
    apptest_with_mocked_agent, monkeypatch, _apply_scratch_files
):
    """End-to-end: approve once, apply once (the deterministic apply step -
    tools.apply_approved_change_set - actually writes the file and marks it
    applied, exactly like apply_approved_change does), the change then
    drops out of the pending list, and a further rerun neither re-approves
    nor re-applies it."""
    at = apptest_with_mocked_agent
    path = _apply_scratch_files("tests/_phase6_ui_scratch_disappear.py")

    import agent as agent_module

    def fake_run_agent_turn_reports(agent, history, **kwargs):
        return {
            "answer": "All tests pass.",
            "tool_calls": [],
            "workflow_states": ["COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_reports)

    change = workflow.register_change(
        file_path=path, action="create", content="x = 1\n", reason="demo"
    )
    workflow._pending_changes.pop(change.change_id)
    change.change_id = "abc123"
    workflow._pending_changes["abc123"] = change

    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change("abc123").applied is True
    assert workflow.get_change("abc123") not in workflow.list_pending_changes()
    # Once applied, re-approving it must be a safe no-op - there is nothing
    # left it should ever be allowed to trigger again.
    assert workflow.approve_change("abc123") is False

    message_count = len(at.session_state["messages"])
    at.run(timeout=30)
    assert at.exception == []
    assert len(at.session_state["messages"]) == message_count
    button_keys = {b.key for b in at.button}
    assert f"approve_set_{change.changeset_id}" not in button_keys
    assert f"apply_set_{change.changeset_id}" not in button_keys


# ---------------------------------------------------------------------------
# Phase 6: final packaging (create_project_zip / the "Download Project ZIP"
# button). The button must only ever appear once a task has genuinely
# reached the real COMPLETED workflow state with a real passing run_pytest -
# never before, and never fabricated - and building the archive must be
# idempotent across Streamlit reruns.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_zip_cache():
    """tools.py's packaging caches (_ZIP_CACHE for the whole-assistant
    archive, _GENERATED_ZIP_CACHE for a generated sub-project's own archive)
    are process-wide module-level stores, the same single-user pattern as
    workflow.py's registries - reset them around every test in this file so
    one test's archive can never make another test's "was it actually
    rebuilt?" assertion see stale state, and so a same-second rebuild in one
    test can never reuse cached bytes for a dist/ file another test already
    cleaned up below."""
    import tools

    tools._ZIP_CACHE.update(signature=None, bytes=None, filename=None, report=None)
    tools._GENERATED_ZIP_CACHE.clear()
    yield
    dist_dir = tools.PROJECT_ROOT / "dist"
    if dist_dir.exists():
        for entry in dist_dir.iterdir():
            if entry.is_file():
                entry.unlink()
    tools._ZIP_CACHE.update(signature=None, bytes=None, filename=None, report=None)


def _set_completed_workflow(agent_module, monkeypatch, *, tests_passed: bool = True):
    """Monkeypatch run_agent_turn to return the real shape agent.py produces
    once a task has finished: a run_pytest tool_call with a genuine
    "Exit code: N" output, plus workflow_states ending in COMPLETED (only
    ever appended by agent.py itself when that exit code was 0 - see
    agent.run_agent_turn)."""
    exit_code = 0 if tests_passed else 1
    states = ["TESTING", "REVIEWING"] + (["COMPLETED"] if tests_passed else ["FAILED"])

    def fake_run_agent_turn(agent, history, **kwargs):
        return {
            "answer": "All done.",
            "tool_calls": [
                {
                    "name": "run_pytest",
                    "input": {},
                    "output": f"Exit code: {exit_code}\n\n1 passed in 0.01s",
                }
            ],
            "workflow_states": states,
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)


def test_download_button_absent_before_any_task(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    download_buttons = [b for b in at.button if b.key == "download_project_zip"]
    assert download_buttons == []


def test_download_button_absent_when_task_not_yet_completed(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    at.run(timeout=30)  # the task strip (and this section) reflect state as of
    assert at.exception == []  # the START of a render - one extra rerun makes
    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert download_buttons == []


def test_download_button_absent_when_tests_failed(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    import agent as agent_module

    _set_completed_workflow(agent_module, monkeypatch, tests_passed=False)
    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert download_buttons == []


def test_download_button_appears_once_workflow_genuinely_completes(
    apptest_with_mocked_agent, monkeypatch
):
    """The just-submitted turn's own render pass never sees this section
    (_current_task() reads state as of the start of that render, before the
    turn it triggers is appended - true for the pre-existing task-progress
    strip too), so a further rerun (e.g. any later interaction/reconnect) is
    what actually reveals it - hence the extra at.run() below."""
    at = apptest_with_mocked_agent
    import agent as agent_module

    _set_completed_workflow(agent_module, monkeypatch, tests_passed=True)
    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert len(download_buttons) == 1
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Project Completed" in markdown_text


def test_download_button_serves_the_real_valid_archive_from_disk(
    apptest_with_mocked_agent, monkeypatch
):
    """Regression coverage for the actual download mechanism, not just the
    button's presence: the exact bytes the button hands to
    st.download_button must be the same archive get_or_build_project_zip
    just wrote to dist/, must open cleanly with Python's own zipfile
    module, and must never contain '.env', '.git', a venv, __pycache__, or
    a previously generated archive."""
    import zipfile

    at = apptest_with_mocked_agent
    import agent as agent_module
    import tools

    _set_completed_workflow(agent_module, monkeypatch, tests_passed=True)

    captured: dict = {}
    real_get_or_build = tools.get_or_build_project_zip

    def capturing_get_or_build(*args, **kwargs):
        package = real_get_or_build(*args, **kwargs)
        captured["package"] = package
        return package

    monkeypatch.setattr(tools, "get_or_build_project_zip", capturing_get_or_build)

    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert len(download_buttons) == 1
    assert "package" in captured

    on_disk = tools.PROJECT_ROOT / "dist" / captured["package"]["filename"]
    assert on_disk.is_file()
    on_disk_bytes = on_disk.read_bytes()
    assert on_disk_bytes == captured["package"]["bytes"]

    with zipfile.ZipFile(on_disk) as zf:
        assert zf.testzip() is None  # every member's CRC actually checks out
        names = [n.lower() for n in zf.namelist()]

    assert not any(n == ".env" or n.endswith("/.env") for n in names)
    assert not any(n.startswith(".git/") for n in names)
    assert not any(n.startswith("venv/") or "/venv/" in n for n in names)
    assert not any("__pycache__" in n for n in names)
    assert not any(n.endswith(".zip") for n in names)


def test_repeated_rerun_after_completion_does_not_rebuild_the_archive(
    apptest_with_mocked_agent, monkeypatch
):
    """Regression coverage for the idempotency requirement: a Streamlit
    rerun after the task already completed must not re-scan/re-zip the
    project again - only the very first render (which built the cache) may
    actually call the real archive builder."""
    at = apptest_with_mocked_agent
    import agent as agent_module

    _set_completed_workflow(agent_module, monkeypatch, tests_passed=True)

    import tools

    real_build = tools._build_zip_bytes
    build_calls = []

    def counting_build(paths):
        build_calls.append(1)
        return real_build(paths)

    monkeypatch.setattr(tools, "_build_zip_bytes", counting_build)

    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    at.run(timeout=30)  # first render that actually sees the completed task
    assert at.exception == []
    assert len(build_calls) == 1

    # A further rerun with no new interaction (e.g. an unrelated widget
    # re-render) must reuse the cached archive rather than rebuilding it.
    at.run(timeout=30)
    assert at.exception == []
    assert len(build_calls) == 1


# ---------------------------------------------------------------------------
# create_pdf / the "Download PDF" button - rendered inline for the turn's
# own create_pdf tool call (unlike the ZIP button, this is never gated on
# the COMPLETED workflow state - a PDF request is a single-tool-call turn).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_generated_files_dir():
    import tools

    output_dir = tools.PROJECT_ROOT / "generated_files"

    def _clear():
        if output_dir.exists():
            for entry in output_dir.iterdir():
                if entry.is_file():
                    entry.unlink()

    _clear()
    yield
    _clear()


def _set_pdf_tool_call(agent_module, monkeypatch, tool_output: str):
    def fake_run_agent_turn(agent, history, **kwargs):
        return {
            "answer": "Here is your PDF.",
            "tool_calls": [
                {
                    "name": "create_pdf",
                    "input": {"title": "Python Basics"},
                    "output": tool_output,
                }
            ],
            "workflow_states": ["PACKAGING"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)


def test_pdf_download_button_appears_after_successful_create_pdf_call(
    apptest_with_mocked_agent, monkeypatch
):
    import agent as agent_module
    import tools

    real_result = tools.create_pdf.invoke(
        {"title": "Python Basics", "content": "# Intro\nPython is great.\n"}
    )
    _set_pdf_tool_call(agent_module, monkeypatch, real_result)

    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("Create a PDF for Python basics").run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key and b.key.startswith("download_pdf_")
    ]
    assert len(download_buttons) == 1


def test_pdf_download_button_serves_the_real_bytes_from_disk(
    apptest_with_mocked_agent, monkeypatch
):
    """Regression coverage for the actual download mechanism, not just the
    button's presence: create_pdf's reported path must point at a real,
    non-empty, valid PDF file - the exact file _render_pdf_download reads
    to build the button (AppTest's DownloadButton element does not expose
    the bytes handed to st.download_button directly - see the file-level
    check here, and test_download_button_serves_the_real_valid_archive_
    from_disk above for the same pattern used with the ZIP button)."""
    import agent as agent_module
    import tools

    real_result = tools.create_pdf.invoke(
        {"title": "Python Basics", "content": "# Intro\nPython is great.\n"}
    )
    rel_path = re.search(r"^Path:\s*(\S+)\s*$", real_result, flags=re.MULTILINE).group(
        1
    )
    on_disk = tools.PROJECT_ROOT / rel_path
    _set_pdf_tool_call(agent_module, monkeypatch, real_result)

    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("Create a PDF for Python basics").run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key and b.key.startswith("download_pdf_")
    ]
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "📄 Download PDF"
    assert on_disk.is_file()
    assert on_disk.read_bytes().startswith(b"%PDF")


def test_pdf_download_button_absent_when_create_pdf_failed(
    apptest_with_mocked_agent, monkeypatch
):
    import agent as agent_module

    _set_pdf_tool_call(
        agent_module,
        monkeypatch,
        "Error: no content was provided to build the PDF from.",
    )

    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("Create a PDF for Python basics").run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key and b.key.startswith("download_pdf_")
    ]
    assert download_buttons == []


def test_pdf_download_button_absent_before_any_pdf_request(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    download_buttons = [
        b for b in at.download_button if b.key and b.key.startswith("download_pdf_")
    ]
    assert download_buttons == []


# ---------------------------------------------------------------------------
# Workflows 1-3: the ACTIVE GENERATED PROJECT hint (so "create documentation
# PDF for this application" never depends on the model recalling a folder
# name), and the task_wants_documentation flag that lets a single combined
# request ("build the app ..., test it, and create documentation PDF")
# survive the human-approval pause between propose and apply.
# ---------------------------------------------------------------------------


def test_active_project_hint_included_when_a_project_is_active(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    at.session_state["active_project_path"] = "generated_projects/todo_app"

    import agent as agent_module

    captured = []

    def fake_run_agent_turn(agent, history, **kwargs):
        captured.append(history[-1].content)
        return {"answer": "ok", "tool_calls": [], "workflow_states": []}

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)

    at.chat_input[0].set_value("Create a documentation PDF for this application.").run(
        timeout=30
    )
    assert at.exception == []

    assert captured
    assert "ACTIVE GENERATED PROJECT: generated_projects/todo_app" in captured[-1]


def test_active_project_hint_absent_when_no_project_generated_yet(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    assert at.session_state["active_project_path"] is None

    import agent as agent_module

    captured = []

    def fake_run_agent_turn(agent, history, **kwargs):
        captured.append(history[-1].content)
        return {"answer": "ok", "tool_calls": [], "workflow_states": []}

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)

    at.chat_input[0].set_value("What is a REST API?").run(timeout=30)
    assert at.exception == []

    assert captured
    assert "ACTIVE GENERATED PROJECT" not in captured[-1]


def test_task_wants_documentation_flag_set_from_combined_request(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "run_agent_turn",
        lambda agent, history, **kwargs: {
            "answer": "ok",
            "tool_calls": [],
            "workflow_states": [],
        },
    )

    at.chat_input[0].set_value(
        "Create a Python Todo app and generate a PDF documentation for the project."
    ).run(timeout=30)
    assert at.exception == []
    assert at.session_state["task_wants_documentation"] is True


def test_task_wants_documentation_flag_false_for_plain_pdf_spec_build(
    apptest_with_mocked_agent, monkeypatch
):
    """A PDF-specification-driven build ("build the app from this PDF")
    mentions "PDF" only as the SOURCE of the requirements - it must not, by
    itself, set the documentation-continuation flag."""
    at = apptest_with_mocked_agent
    import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "run_agent_turn",
        lambda agent, history, **kwargs: {
            "answer": "ok",
            "tool_calls": [],
            "workflow_states": [],
        },
    )

    at.chat_input[0].set_value("Build the application based on this PDF.").run(
        timeout=30
    )
    assert at.exception == []
    assert at.session_state["task_wants_documentation"] is False


def test_also_generate_documentation_passed_through_to_run_agent_turn(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    import agent as agent_module

    captured_kwargs = []

    def fake_run_agent_turn(agent, history, **kwargs):
        captured_kwargs.append(kwargs)
        return {"answer": "ok", "tool_calls": [], "workflow_states": []}

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)

    at.chat_input[0].set_value(
        "Create a Todo app and generate documentation PDF for it."
    ).run(timeout=30)
    assert at.exception == []
    assert captured_kwargs[-1].get("also_generate_documentation") is True


def test_apply_resume_nudge_mentions_documentation_when_task_wants_it(
    apptest_with_mocked_agent, monkeypatch, _apply_scratch_files
):
    """The core Workflow 3 regression test: task_wants_documentation set on
    the original request must still be honored after the human-approval
    pause (Approve, then Apply) - the resume nudge sent to run_agent_turn
    must explicitly ask for the documentation PDF too."""
    at = apptest_with_mocked_agent
    path = _apply_scratch_files("tests/_phase6_ui_scratch_doc_resume.py")

    import agent as agent_module

    at.session_state["task_wants_documentation"] = True

    calls = []

    def fake_run_agent_turn_resuming(agent, history, **kwargs):
        calls.append(history[-1].content)
        return {
            "answer": "Applied the change, tests pass, and documentation PDF created.",
            "tool_calls": [],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_resuming)

    change = workflow.register_change(
        file_path=path, action="create", content="x = 1\n", reason="demo"
    )
    workflow._pending_changes.pop(change.change_id)
    change.change_id = "docresume1"
    workflow._pending_changes["docresume1"] = change

    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert calls, "run_agent_turn should have been called to resume the workflow"
    assert "documentation PDF" in calls[-1]
    assert "create_pdf" in calls[-1]


def test_apply_resume_nudge_omits_documentation_clause_when_not_requested(
    apptest_with_mocked_agent, monkeypatch, _apply_scratch_files
):
    at = apptest_with_mocked_agent
    path = _apply_scratch_files("tests/_phase6_ui_scratch_no_doc_resume.py")

    import agent as agent_module

    at.session_state["task_wants_documentation"] = False

    calls = []

    def fake_run_agent_turn_resuming(agent, history, **kwargs):
        calls.append(history[-1].content)
        return {
            "answer": "Applied the change and all tests pass.",
            "tool_calls": [],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_resuming)

    change = workflow.register_change(
        file_path=path, action="create", content="x = 1\n", reason="demo"
    )
    workflow._pending_changes.pop(change.change_id)
    change.change_id = "nodocresume1"
    workflow._pending_changes["nodocresume1"] = change

    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert calls
    assert "documentation PDF" not in calls[-1]


# ---------------------------------------------------------------------------
# Phase 6: Live Application Preview (launch_generated_app/stop_generated_app)
# - the "🚀 Live Application" section and its Start/Stop buttons.
# ---------------------------------------------------------------------------

_GENERATED_PROJECT_ROOT = "generated_projects/_apptest_todo"


@pytest.fixture(autouse=True)
def _clean_generated_server_state():
    import tools

    tools._GENERATED_SERVERS.clear()
    yield
    tools._GENERATED_SERVERS.clear()


@pytest.fixture
def scratch_generated_app_dir():
    """A real, disposable generated Streamlit project on disk (real
    launch_generated_app validation reads the actual file, not just the
    workflow registry) - removed after the test regardless of outcome."""
    import tools

    root = tools.PROJECT_ROOT / _GENERATED_PROJECT_ROOT
    root.mkdir(parents=True)
    (root / "app.py").write_text("import streamlit as st\nst.write('hi')\n")
    yield root
    import shutil

    shutil.rmtree(root, ignore_errors=True)


def _apply_fake_generated_change(filename: str = "app.py") -> str:
    change = workflow.register_change(
        file_path=f"{_GENERATED_PROJECT_ROOT}/{filename}",
        action="create",
        content="import streamlit as st\n",
        reason="generated app (test)",
    )
    workflow.approve_change(change.change_id)
    workflow.mark_applied(change.change_id)
    return change.change_id


def _set_generated_app_verified(agent_module, monkeypatch, change_id: str) -> None:
    """Mocks run_agent_turn to return the real shape agent.py produces once
    a generated project's own tests have passed: an apply_approved_change
    call for the generated file, and a run_pytest call whose `target`
    targets that exact generated project, with a genuine passing exit code."""

    def fake_run_agent_turn(agent, history, **kwargs):
        return {
            "answer": "Done.",
            "tool_calls": [
                {
                    "name": "apply_approved_change",
                    "input": {"change_id": change_id},
                    "output": "ok",
                },
                {
                    "name": "run_pytest",
                    "input": {"target": f"{_GENERATED_PROJECT_ROOT}/tests"},
                    "output": "Exit code: 0\n\n2 passed in 0.02s",
                },
            ],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn)


def test_live_application_section_absent_before_generated_tests_pass(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Live Application" not in markdown_text


def test_start_live_app_button_appears_once_generated_tests_pass(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent
    import agent as agent_module

    change_id = _apply_fake_generated_change()
    _set_generated_app_verified(agent_module, monkeypatch, change_id)

    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Live Application" in markdown_text
    start_buttons = [b for b in at.button if "Start Live App" in b.label]
    assert len(start_buttons) == 1


def test_clicking_start_live_app_launches_exactly_once(
    apptest_with_mocked_agent, monkeypatch, scratch_generated_app_dir
):
    at = apptest_with_mocked_agent
    import subprocess

    import agent as agent_module
    import tools

    change_id = _apply_fake_generated_change()
    _set_generated_app_verified(agent_module, monkeypatch, change_id)

    launch_calls = []

    class _FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        launch_calls.append(args)
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)

    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    start_btn = next(b for b in at.button if "Start Live App" in b.label)
    start_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert len(launch_calls) == 1
    assert tools.get_generated_server(_GENERATED_PROJECT_ROOT).status == "running"
    assert "http://localhost:8502" in "\n".join(c.value for c in at.caption)
    # st.link_button isn't a typed AppTest element in this Streamlit version -
    # it comes back as an UnknownElement whose raw proto still carries the
    # real label/url, which is exactly what matters here (a real tracked
    # URL, never one built from model output).
    link_elements = at.get("link_button")
    assert len(link_elements) == 1
    assert link_elements[0].proto.label == "Open Live App"
    assert link_elements[0].proto.url == "http://localhost:8502"

    # A further rerun with no new click must not launch a second time.
    at.run(timeout=30)
    assert at.exception == []
    assert len(launch_calls) == 1


def test_stop_app_button_stops_and_shows_not_running(
    apptest_with_mocked_agent, monkeypatch, scratch_generated_app_dir
):
    at = apptest_with_mocked_agent
    import subprocess

    import agent as agent_module
    import tools

    change_id = _apply_fake_generated_change()
    _set_generated_app_verified(agent_module, monkeypatch, change_id)

    class _FakeProcess:
        pid = 12345
        terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    fake_process = _FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: fake_process)
    monkeypatch.setattr(tools, "_select_safe_port", lambda: 8502)
    monkeypatch.setattr(tools, "_http_health_check", lambda port: True)
    # Never invoke the real taskkill/killpg boundary against this fake PID.
    monkeypatch.setattr(
        tools,
        "_terminate_process_tree",
        lambda pid, process=None: process.terminate() if process else None,
    )

    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    start_btn = next(b for b in at.button if "Start Live App" in b.label)
    start_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert tools.get_generated_server(_GENERATED_PROJECT_ROOT).status == "running"

    stop_btn = next(b for b in at.button if "Stop App" in b.label)
    stop_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert tools.get_generated_server(_GENERATED_PROJECT_ROOT).status == "stopped"
    assert fake_process.terminated is True
    markdown_text = "\n".join(m.value for m in at.markdown)
    assert "Not running" in markdown_text
    assert [b for b in at.button if "Stop App" in b.label] == []


def test_live_application_section_coexists_with_zip_download(
    apptest_with_mocked_agent, monkeypatch, scratch_generated_app_dir
):
    """Adding the live-preview section must never break the existing ZIP
    download button - both can be shown for the same completed task."""
    at = apptest_with_mocked_agent
    import agent as agent_module

    change_id = _apply_fake_generated_change()
    _set_generated_app_verified(agent_module, monkeypatch, change_id)

    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert len(download_buttons) == 1


def test_zip_download_for_generated_project_packages_only_that_project(
    apptest_with_mocked_agent, monkeypatch, scratch_generated_app_dir
):
    """Once a task actually built a generated sub-project (e.g. a to-do
    app), the downloadable archive must contain ONLY that sub-project's own
    files - never this whole AI Developer Assistant repository. Regression
    coverage for a real bug: the download button used to always call
    get_or_build_project_zip() (the whole-assistant archive) even when the
    completed task was a generated sub-project.

    Follows the same capture-and-verify pattern as
    test_download_button_serves_the_real_valid_archive_from_disk: the AppTest
    DownloadButton element's proto only carries a deferred_file_id handle
    (the real bytes live in Streamlit's own media file storage), so the
    actual archive is verified via the real get_or_build_generated_project_zip
    call and the real file it wrote to dist/.
    """
    import zipfile

    at = apptest_with_mocked_agent
    import agent as agent_module
    import tools

    _set_generated_app_verified(
        agent_module, monkeypatch, _apply_fake_generated_change()
    )

    captured: dict = {}
    real_get_or_build = tools.get_or_build_generated_project_zip

    def capturing_get_or_build(*args, **kwargs):
        package = real_get_or_build(*args, **kwargs)
        captured["package"] = package
        return package

    monkeypatch.setattr(
        tools, "get_or_build_generated_project_zip", capturing_get_or_build
    )

    at.chat_input[0].set_value("Build a to-do app").run(timeout=30)
    at.run(timeout=30)
    assert at.exception == []

    download_buttons = [
        b for b in at.download_button if b.key == "download_project_zip"
    ]
    assert len(download_buttons) == 1
    assert "package" in captured
    package = captured["package"]

    assert package["filename"] == "apptest_todo.zip"
    assert package["files"] == ["_apptest_todo/app.py"]
    assert not any(f.startswith("generated_projects/") for f in package["files"])
    assert "tools.py" not in package["files"]
    assert "workflow.py" not in package["files"]

    on_disk = tools.PROJECT_ROOT / "dist" / package["filename"]
    assert on_disk.is_file()
    assert on_disk.read_bytes() == package["bytes"]
    with zipfile.ZipFile(io.BytesIO(package["bytes"])) as archive:
        assert archive.namelist() == ["_apptest_todo/app.py"]


def test_zip_scopes_to_generated_project_when_applied_via_apply_button(
    apptest_with_mocked_agent, monkeypatch, scratch_generated_app_dir
):
    """Regression test for a real bug: a generated project's proposed files
    are normally written to disk via this UI's own "Apply Approved Changes"
    button (tools.apply_approved_change_set), NOT via the agent calling the
    apply_approved_change tool itself - but _current_generated_project_root()
    used to look ONLY for an "apply_approved_change" tool_call in chat
    history, which the button path never produces. That made the download
    button silently fall back to get_or_build_project_zip() (this whole
    assistant repository) for every project actually built through the real
    Approve/Apply UI flow, even though a chat-driven apply (as covered by
    test_zip_download_for_generated_project_packages_only_that_project) was
    scoped correctly. Uses the real Approve/Apply buttons end to end, never a
    fabricated apply_approved_change tool_call, to catch exactly this gap.
    """
    at = apptest_with_mocked_agent
    import agent as agent_module
    import tools

    change = workflow.register_change(
        file_path=f"{_GENERATED_PROJECT_ROOT}/app.py",
        action="modify",
        content="import streamlit as st\nst.write('hi there')\n",
        reason="generated app (test)",
    )

    def fake_run_agent_turn_after_apply(agent, history, **kwargs):
        return {
            "answer": "Tests pass.",
            "tool_calls": [
                {
                    "name": "run_pytest",
                    "input": {"target": f"{_GENERATED_PROJECT_ROOT}/tests"},
                    "output": "Exit code: 0\n\n2 passed in 0.02s",
                },
            ],
            "workflow_states": ["IMPLEMENTING", "TESTING", "REVIEWING", "COMPLETED"],
        }

    monkeypatch.setattr(agent_module, "run_agent_turn", fake_run_agent_turn_after_apply)

    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert workflow.get_change(change.change_id).applied is True

    captured: dict = {}
    real_get_or_build = tools.get_or_build_generated_project_zip

    def capturing_get_or_build(*args, **kwargs):
        package = real_get_or_build(*args, **kwargs)
        captured["package"] = package
        return package

    monkeypatch.setattr(
        tools, "get_or_build_generated_project_zip", capturing_get_or_build
    )

    at = _goto(at, "Chat")
    assert "package" in captured
    package = captured["package"]

    assert package["files"] == ["_apptest_todo/app.py"]
    assert not any(f.startswith("generated_projects/") for f in package["files"])
    assert "tools.py" not in package["files"]
    assert "workflow.py" not in package["files"]
    assert "agent.py" not in package["files"]


# ---------------------------------------------------------------------------
# Project tab - shows ONLY the explicitly tracked active generated project
# (st.session_state.active_project_path), never every folder under
# generated_projects/. See app.py's _note_applied_change/render_project_page.
# ---------------------------------------------------------------------------


@pytest.fixture
def two_scratch_generated_projects():
    """Two real, disposable generated projects on disk at once, so tests
    can prove the Project tab shows only the active one and never both."""
    import tools

    first_root = tools.PROJECT_ROOT / "generated_projects" / "_apptest_first"
    second_root = tools.PROJECT_ROOT / "generated_projects" / "_apptest_second"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    (first_root / "app.py").write_text("import streamlit as st\n")
    (second_root / "main.py").write_text("import streamlit as st\n")
    yield first_root, second_root
    import shutil

    shutil.rmtree(first_root, ignore_errors=True)
    shutil.rmtree(second_root, ignore_errors=True)


def _approve_and_apply_via_ui(at, change: workflow.ProposedChange):
    """Drive the real Approve then Apply buttons on the Changes page for one
    change set - the same real UI path a generated project's files are
    actually written to disk through (see
    test_zip_scopes_to_generated_project_when_applied_via_apply_button),
    never a direct workflow.mark_applied() shortcut - so this exercises the
    real app.py code (_apply_changeset_and_resume -> _note_applied_change)
    that sets st.session_state.active_project_path."""
    at = _goto(at, "Changes")
    approve_btn = next(
        b for b in at.button if b.key == f"approve_set_{change.changeset_id}"
    )
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    apply_btn = next(
        b for b in at.button if b.key == f"apply_set_{change.changeset_id}"
    )
    apply_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert workflow.get_change(change.change_id).applied is True
    return at


def test_project_tab_shows_no_project_selected_message_before_any_generation(
    apptest_with_mocked_agent,
):
    at = _goto(apptest_with_mocked_agent, "Project")
    caption_text = "\n".join(c.value for c in at.caption)
    assert (
        "No project selected. Generate or select a project to view its "
        "structure." in caption_text
    )
    code_blocks = [c.value for c in at.code]
    assert code_blocks == []


def test_project_tab_shows_only_the_active_generated_project(
    apptest_with_mocked_agent, two_scratch_generated_projects
):
    at = apptest_with_mocked_agent
    first_root, _second_root = two_scratch_generated_projects

    change = workflow.register_change(
        file_path="generated_projects/_apptest_first/app.py",
        action="create",
        content="import streamlit as st\n",
        reason="generated app (test)",
    )
    at = _approve_and_apply_via_ui(at, change)

    at = _goto(at, "Project")
    code_text = "\n".join(c.value for c in at.code)
    assert "_apptest_first" in code_text
    assert "app.py" in code_text
    assert "_apptest_second" not in code_text
    assert "main.py" not in code_text


def test_project_tab_switches_to_the_newly_generated_project(
    apptest_with_mocked_agent, two_scratch_generated_projects
):
    """Generating project A then project B must flip the Project tab from
    showing only A to showing only B - never both at once."""
    at = apptest_with_mocked_agent
    first_root, second_root = two_scratch_generated_projects

    change_a = workflow.register_change(
        file_path="generated_projects/_apptest_first/app.py",
        action="create",
        content="import streamlit as st\n",
        reason="generated app (test)",
    )
    at = _approve_and_apply_via_ui(at, change_a)

    at = _goto(at, "Project")
    code_text = "\n".join(c.value for c in at.code)
    assert "_apptest_first" in code_text
    assert "_apptest_second" not in code_text

    change_b = workflow.register_change(
        file_path="generated_projects/_apptest_second/main.py",
        action="create",
        content="import streamlit as st\n",
        reason="generated app (test)",
    )
    at = _approve_and_apply_via_ui(at, change_b)

    at = _goto(at, "Project")
    code_text = "\n".join(c.value for c in at.code)
    assert "_apptest_second" in code_text
    assert "main.py" in code_text
    assert "_apptest_first" not in code_text
    assert "app.py" not in code_text


def test_project_tab_refresh_still_scopes_to_the_active_project(
    apptest_with_mocked_agent, two_scratch_generated_projects
):
    at = apptest_with_mocked_agent
    first_root, _second_root = two_scratch_generated_projects

    change = workflow.register_change(
        file_path="generated_projects/_apptest_first/app.py",
        action="create",
        content="import streamlit as st\n",
        reason="generated app (test)",
    )
    at = _approve_and_apply_via_ui(at, change)

    at = _goto(at, "Project")
    refresh_btn = next(b for b in at.button if b.key == "refresh_project_files")
    refresh_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    code_text = "\n".join(c.value for c in at.code)
    assert "_apptest_first" in code_text
    assert "_apptest_second" not in code_text
