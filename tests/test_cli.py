"""
Tests for cli.py's Phase 6 approval flow.

The CLI has no sidebar Approve/Reject buttons like app.py, so it must ask
for approval directly at the terminal instead of leaving the user stuck at
WAITING_FOR_APPROVAL with no way to ever get past it. These tests drive
`_handle_pending_approvals`/`_run_turn` directly against the real
`workflow.py` registry (never a real Gemini call - `run_agent_turn` is
monkeypatched at the module level, the same pattern test_agent.py uses for
`ask_agent`/`classify_request`).
"""

import io

import pytest

import cli as cli_module
import workflow
from cli import (
    _MAX_APPROVAL_ROUNDS,
    _handle_pending_approvals,
    _print_pending_change,
    _run_turn,
    main,
)


class _FakeAgent:
    pass


@pytest.fixture(autouse=True)
def _clean_registry():
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()
    yield
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()


# ---------------------------------------------------------------------------
# _handle_pending_approvals
# ---------------------------------------------------------------------------


def test_handle_pending_approvals_approves_when_user_says_yes(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    approved = _handle_pending_approvals([change.change_id])

    assert approved == [change.change_id]
    assert workflow.get_change(change.change_id).approved is True


def test_handle_pending_approvals_rejects_when_user_says_no(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    approved = _handle_pending_approvals([change.change_id])

    assert approved == []
    assert workflow.get_change(change.change_id) is None  # rejected removes it


def test_handle_pending_approvals_defaults_to_reject_on_empty_input(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    approved = _handle_pending_approvals([change.change_id])

    assert approved == []


def test_handle_pending_approvals_skips_unknown_change_id(monkeypatch):
    def _fail_if_called(prompt=""):
        raise AssertionError("must never prompt for an id that isn't a real change")

    monkeypatch.setattr("builtins.input", _fail_if_called)

    approved = _handle_pending_approvals(["not-a-real-id"])

    assert approved == []


def test_handle_pending_approvals_skips_already_applied_change(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    workflow.approve_change(change.change_id)
    workflow.mark_applied(change.change_id)

    def _fail_if_called(prompt=""):
        raise AssertionError("must never re-prompt for an already-applied change")

    monkeypatch.setattr("builtins.input", _fail_if_called)

    approved = _handle_pending_approvals([change.change_id])

    assert approved == []


def test_handle_pending_approvals_treats_keyboard_interrupt_as_reject(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")

    def _raise_interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)

    approved = _handle_pending_approvals([change.change_id])

    assert approved == []
    assert workflow.get_change(change.change_id) is None


# ---------------------------------------------------------------------------
# _run_turn
# ---------------------------------------------------------------------------


def test_run_turn_returns_answer_directly_when_nothing_is_pending(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "run_agent_turn",
        lambda agent, conversation: {"answer": "done", "pending_change_ids": []},
    )

    conversation = []
    answer = _run_turn(_FakeAgent(), conversation)

    assert answer == "done"
    assert len(conversation) == 1  # just the AI reply was recorded


def test_run_turn_does_not_crash_on_a_legacy_console_codepage(monkeypatch):
    """Regression test: a real Gemini answer containing an emoji (folder
    icon) crashed a bare print() with UnicodeEncodeError on a simulated
    legacy Windows console codepage (cp1252) - _run_turn must use
    logger.safe_print, not print(), for the "Assistant: ..." line so an
    emoji in the model's own answer can never kill the CLI session."""
    answer_with_emoji = "Here is the ### \U0001f4c1 Project Structure"
    monkeypatch.setattr(
        cli_module,
        "run_agent_turn",
        lambda agent, conversation: {
            "answer": answer_with_emoji,
            "pending_change_ids": [],
        },
    )

    buffer = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", newline=""
    )
    monkeypatch.setattr("sys.stdout", buffer)

    answer = _run_turn(_FakeAgent(), [])  # must not raise UnicodeEncodeError
    buffer.flush()

    assert answer == answer_with_emoji
    buffer.seek(0)
    printed = buffer.buffer.getvalue().decode("cp1252")
    assert "Assistant:" in printed
    assert "?" in printed  # the emoji was replaced, not left to crash printing


def test_print_pending_change_does_not_crash_on_a_legacy_console_codepage(
    monkeypatch,
):
    """Regression test for a real crash found in live testing: a proposed
    file's own content (e.g. a generated Streamlit app that legitimately
    uses emoji in its UI copy) crashed the ENTIRE CLI session with
    UnicodeEncodeError on a legacy Windows console codepage (cp1252) -
    _print_pending_change used a bare print(change.content) instead of
    logger.safe_print, the exact bug already fixed for the "Assistant: ..."
    line (see test_run_turn_does_not_crash_on_a_legacy_console_codepage)
    but missed here."""
    change = workflow.register_change(
        file_path="generated_projects/demo/app.py",
        action="create",
        content='st.write("\U0001f4dd Notes")',
        reason="Add a \U0001f4dd notes section",
    )

    buffer = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict", newline=""
    )
    monkeypatch.setattr("sys.stdout", buffer)

    _print_pending_change(change)  # must not raise UnicodeEncodeError
    buffer.flush()

    buffer.seek(0)
    printed = buffer.buffer.getvalue().decode("cp1252")
    assert "APPROVAL REQUIRED" in printed
    assert change.file_path in printed
    assert "?" in printed  # the emoji was replaced, not left to crash printing


def test_run_turn_prompts_for_approval_and_continues_once_approved(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    calls = []

    def fake_run_agent_turn(agent, conversation):
        calls.append(conversation)
        if len(calls) == 1:
            return {
                "answer": "Proposed a change - please approve.",
                "pending_change_ids": [change.change_id],
            }
        return {"answer": "Applied and tests passed.", "pending_change_ids": []}

    monkeypatch.setattr(cli_module, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    conversation = []
    answer = _run_turn(_FakeAgent(), conversation)

    assert len(calls) == 2  # the initial proposal, then the approved follow-up
    assert answer == "Applied and tests passed."
    assert workflow.get_change(change.change_id).approved is True


def test_run_turn_does_not_continue_when_the_change_is_rejected(monkeypatch):
    change = workflow.register_change("a.py", "create", "x = 1\n", "demo")
    calls = []

    def fake_run_agent_turn(agent, conversation):
        calls.append(conversation)
        return {
            "answer": "Proposed a change - please approve.",
            "pending_change_ids": [change.change_id],
        }

    monkeypatch.setattr(cli_module, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    conversation = []
    _run_turn(_FakeAgent(), conversation)

    assert len(calls) == 1  # never asked the agent to apply a rejected change


def test_run_turn_never_exceeds_max_approval_rounds(monkeypatch):
    """Never loop forever: even if the model kept proposing a fresh change
    every round, the CLI must eventually stop asking."""
    calls = []

    def fake_run_agent_turn(agent, conversation):
        calls.append(conversation)
        change = workflow.register_change(
            f"file_{len(calls)}.py", "create", "x = 1\n", "demo"
        )
        return {
            "answer": f"Proposed change {change.change_id}.",
            "pending_change_ids": [change.change_id],
        }

    monkeypatch.setattr(cli_module, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    conversation = []
    _run_turn(_FakeAgent(), conversation)

    assert len(calls) == _MAX_APPROVAL_ROUNDS + 1


# ---------------------------------------------------------------------------
# main()'s API key gate - a genuinely missing key must still stop with a
# clear message, but a key that merely doesn't match the traditional
# "AIza..." shape must not block startup (only a real rejection from
# Google's API should ever be treated as proof a key is invalid).
# ---------------------------------------------------------------------------

_FAKE_INVALID_FORMAT_KEY = "AQ.FakeNonGeminiTokenForTestingOnly1234567890"


def test_main_stops_with_clear_message_when_api_key_missing(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    main()
    assert "not configured" in capsys.readouterr().out


def test_main_proceeds_past_a_differently_shaped_key(monkeypatch, capsys):
    """A key shaped like the real 'AQ.'-prefixed credential from a past
    support case must not stop the CLI - it must reach build_agent()
    (mocked here so no real network call is made)."""
    monkeypatch.setenv("GOOGLE_API_KEY", _FAKE_INVALID_FORMAT_KEY)
    monkeypatch.setattr(cli_module, "build_agent", lambda: _FakeAgent())
    monkeypatch.setattr(
        "builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError())
    )

    main()

    out = capsys.readouterr().out
    assert "does not look like a valid gemini api key" not in out.lower()
    assert "CLI mode" in out
    assert _FAKE_INVALID_FORMAT_KEY not in out
