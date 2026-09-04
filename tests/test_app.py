"""
Tests for app.py's document-attachment and voice-input UI wiring.

These drive the real Streamlit script headlessly via streamlit.testing.v1's
AppTest (no browser, no real Gemini/network calls - agent.ask_agent and
agent.transcribe_audio are monkeypatched). They exist to protect attachment
state, clearing attachments, upload rejection, voice-to-text reaching the
existing chat flow, and that the AI Features sidebar panel is informational
only (no buttons, no auto-inserted or auto-submitted questions).
"""

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
    """Run app.py headlessly with ask_agent/transcribe_audio replaced by
    fakes, so no real API key or network call is ever needed."""
    import agent as agent_module

    def fake_ask_agent(agent, history):
        last = history[-1]
        text = getattr(last, "content", str(last))
        return {"answer": f"MOCKED REPLY for: {text[:200]}", "tool_calls": []}

    def fake_transcribe_audio(audio_bytes, mime_type="audio/wav"):
        return "explain how tools.py works in my project"

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent)
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
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["attached_files"] != []

    clear_btn = next(b for b in at.sidebar.button if b.label == "🗑 Clear Attachments")
    clear_btn.click()
    at.run(timeout=30)
    assert at.exception == []
    assert at.session_state["attached_files"] == []


def test_clearing_attachments_does_not_touch_conversation(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    from documents import process_upload

    # Send one normal message first.
    at.chat_input[0].set_value("What is Python?").run(timeout=30)
    assert at.exception == []
    message_count_before = len(at.session_state["messages"])
    assert message_count_before > 0

    at.session_state["attached_files"] = [process_upload("a.py", b"x = 1\n")]
    at.run(timeout=30)
    clear_btn = next(b for b in at.sidebar.button if b.label == "🗑 Clear Attachments")
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


def test_attaching_file_with_question_in_one_submission_updates_sidebar_same_turn(
    apptest_with_mocked_agent, monkeypatch
):
    # Regression test: the "Attached Files" sidebar section is rendered
    # earlier in the script than the chat_input handling code, so without an
    # internal rerun a newly attached file would only appear in the sidebar
    # on the *next* interaction, not the one that attached it. Confirms
    # session_state (what the sidebar reads) is updated within this same
    # at.run() call, immediately after a combined attach+ask submission.
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


def test_ai_features_panel_lists_all_five_phases_with_no_buttons(
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
    # Information only: no buttons inside a phase's reference panel (the
    # only sidebar buttons left are Clear Attachments / Clear Conversation).
    assert len(phase1.button) == 0
    phase1_text = "\n".join(m.value for m in phase1.markdown)
    assert "Calculator" in phase1_text
    assert "Example capability" in phase1_text


def test_expanding_ai_features_panel_does_not_touch_chat_or_pending_prompt(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent

    messages_before = list(at.session_state["messages"])
    assert at.session_state["pending_prompt"] is None

    # Merely having the AI Features expanders rendered (as they are on every
    # run of this fixture) must never insert, select, or submit a question.
    at.run(timeout=30)
    assert at.exception == []

    assert at.session_state["messages"] == messages_before
    assert at.session_state["pending_prompt"] is None


# ---------------------------------------------------------------------------
# Phase 6: Pending Approvals sidebar (propose -> human approve/reject -> apply)
# ---------------------------------------------------------------------------


def test_no_pending_changes_shows_empty_state(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    markdown_text = "\n".join(m.value for m in at.sidebar.markdown)
    caption_text = "\n".join(c.value for c in at.sidebar.caption)
    assert "No pending changes" in markdown_text + caption_text


def test_pending_change_appears_in_sidebar_with_approve_reject_buttons(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)
    assert at.exception == []

    expander_labels = [e.label for e in at.sidebar.expander]
    assert any(change.change_id in label for label in expander_labels)

    matching_expander = next(
        e for e in at.sidebar.expander if change.change_id in e.label
    )
    button_labels = {b.label for b in matching_expander.button}
    assert "✅ Approve" in button_labels
    assert "❌ Reject" in button_labels


def test_clicking_approve_sets_change_approved(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)

    matching_expander = next(
        e for e in at.sidebar.expander if change.change_id in e.label
    )
    approve_btn = next(b for b in matching_expander.button if b.label == "✅ Approve")
    approve_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id).approved is True


def test_clicking_reject_removes_pending_change(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)

    matching_expander = next(
        e for e in at.sidebar.expander if change.change_id in e.label
    )
    reject_btn = next(b for b in matching_expander.button if b.label == "❌ Reject")
    reject_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.get_change(change.change_id) is None
    assert workflow.list_pending_changes() == []


def test_approving_a_change_does_not_touch_conversation_or_other_state(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    at.chat_input[0].set_value("What is Python?").run(timeout=30)
    messages_before = list(at.session_state["messages"])

    change = workflow.register_change(
        file_path="demo.py", action="create", content="x = 1\n", reason="demo"
    )
    at.run(timeout=30)
    matching_expander = next(
        e for e in at.sidebar.expander if change.change_id in e.label
    )
    approve_btn = next(b for b in matching_expander.button if b.label == "✅ Approve")
    approve_btn.click()
    at.run(timeout=30)

    assert at.session_state["messages"] == messages_before


def test_workflow_states_are_displayed_when_present(
    apptest_with_mocked_agent, monkeypatch
):
    at = apptest_with_mocked_agent

    import agent as agent_module

    def fake_ask_agent_with_workflow(agent, history):
        return {
            "answer": "Done.",
            "tool_calls": [],
            "workflow_states": ["IMPLEMENTING", "WAITING_FOR_APPROVAL"],
        }

    monkeypatch.setattr(agent_module, "ask_agent", fake_ask_agent_with_workflow)
    at.chat_input[0].set_value("Add a feature").run(timeout=30)
    assert at.exception == []

    last_message = at.session_state["messages"][-1]
    assert last_message["workflow_states"] == ["IMPLEMENTING", "WAITING_FOR_APPROVAL"]


# ---------------------------------------------------------------------------
# Phase 6: repair-attempt circuit breaker reset control
# ---------------------------------------------------------------------------


def test_no_repair_warning_when_no_file_has_hit_the_limit(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    warning_text = "\n".join(w.value for w in at.sidebar.warning)
    assert "repair-attempt limit" not in warning_text.lower()
    reset_buttons = [b for b in at.sidebar.button if "Reset repair counter" in b.label]
    assert reset_buttons == []


def test_repair_warning_and_reset_button_appear_once_limit_reached(
    apptest_with_mocked_agent,
):
    at = apptest_with_mocked_agent
    for _ in range(workflow.MAX_REPAIR_ATTEMPTS):
        workflow.record_apply("stuck_file.py")
    at.run(timeout=30)
    assert at.exception == []

    warning_text = "\n".join(w.value for w in at.sidebar.warning)
    assert "repair-attempt limit" in warning_text.lower()
    assert "stuck_file.py" in warning_text

    reset_buttons = [b for b in at.sidebar.button if "Reset repair counter" in b.label]
    assert len(reset_buttons) == 1


def test_clicking_reset_repair_counter_clears_the_tally(apptest_with_mocked_agent):
    at = apptest_with_mocked_agent
    for _ in range(workflow.MAX_REPAIR_ATTEMPTS):
        workflow.record_apply("stuck_file.py")
    at.run(timeout=30)

    reset_btn = next(b for b in at.sidebar.button if "Reset repair counter" in b.label)
    reset_btn.click()
    at.run(timeout=30)
    assert at.exception == []

    assert workflow.repair_attempts_for("stuck_file.py") == 0
