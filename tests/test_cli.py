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

import pytest

import cli as cli_module
import workflow
from cli import _MAX_APPROVAL_ROUNDS, _handle_pending_approvals, _run_turn


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
