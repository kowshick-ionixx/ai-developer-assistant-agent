"""
Tests for workflow.py: WorkflowState/WorkflowStatus transitions and the
pending-change approval registry (the human-approval gate behind
tools.py's propose_file_change/apply_approved_change).
"""

import pytest

from workflow import (
    MAX_REPAIR_ATTEMPTS,
    WorkflowState,
    WorkflowStatus,
    approve_change,
    clear_all_changes,
    get_change,
    list_pending_changes,
    mark_applied,
    record_apply,
    record_test_outcome,
    register_change,
    reject_change,
    repair_attempts_for,
    repair_attempts_snapshot,
    repair_limit_reached,
    reset_repair_attempts,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test gets an empty pending-change registry, since it's a
    process-wide module-level store."""
    clear_all_changes()
    reset_repair_attempts()
    yield
    clear_all_changes()
    reset_repair_attempts()


# ---------------------------------------------------------------------------
# WorkflowState / WorkflowStatus transitions
# ---------------------------------------------------------------------------


def test_initial_status_is_idle():
    state = WorkflowState(user_task="Add a login API")
    assert state.status == WorkflowStatus.IDLE


def test_valid_transition_sequence():
    state = WorkflowState(user_task="Add a feature")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.INSPECTING)
    state.transition_to(WorkflowStatus.IMPLEMENTING)
    state.transition_to(WorkflowStatus.TESTING)
    state.transition_to(WorkflowStatus.REVIEWING)
    state.transition_to(WorkflowStatus.COMPLETED)
    assert state.status == WorkflowStatus.COMPLETED


def test_same_state_transition_is_a_noop():
    state = WorkflowState(user_task="Fix a bug")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.PLANNING)
    assert state.status == WorkflowStatus.PLANNING


def test_invalid_transition_raises_and_does_not_change_state():
    state = WorkflowState(user_task="Refactor code")
    state.transition_to(WorkflowStatus.PLANNING)
    with pytest.raises(ValueError):
        state.transition_to(WorkflowStatus.COMPLETED)
    assert state.status == WorkflowStatus.PLANNING


def test_terminal_states_allow_no_further_transitions():
    state = WorkflowState(user_task="Generate tests")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.FAILED)
    with pytest.raises(ValueError):
        state.transition_to(WorkflowStatus.PLANNING)


def test_repair_loop_can_reach_waiting_for_approval_and_retesting():
    state = WorkflowState(user_task="Add a feature")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.INSPECTING)
    state.transition_to(WorkflowStatus.IMPLEMENTING)
    state.transition_to(WorkflowStatus.TESTING)
    state.transition_to(WorkflowStatus.ANALYZING)
    state.transition_to(WorkflowStatus.FIXING)
    state.transition_to(WorkflowStatus.RETESTING)
    state.transition_to(WorkflowStatus.REVIEWING)
    state.transition_to(WorkflowStatus.COMPLETED)
    assert state.status == WorkflowStatus.COMPLETED


def test_full_propose_to_completed_sequence_including_new_states():
    """Regression test: PROPOSING_CHANGE and REGRESSION_TESTING must exist
    and be reachable as part of the full Phase 6 sequence - a request to add
    these states to the state machine (they were previously missing)."""
    state = WorkflowState(user_task="Add a feature")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.INSPECTING)
    state.transition_to(WorkflowStatus.PROPOSING_CHANGE)
    state.transition_to(WorkflowStatus.WAITING_FOR_APPROVAL)
    state.transition_to(WorkflowStatus.IMPLEMENTING)
    state.transition_to(WorkflowStatus.TESTING)
    state.transition_to(WorkflowStatus.REGRESSION_TESTING)
    state.transition_to(WorkflowStatus.REVIEWING)
    state.transition_to(WorkflowStatus.COMPLETED)
    assert state.status == WorkflowStatus.COMPLETED


def test_repair_loop_can_reach_proposing_change_and_regression_testing():
    state = WorkflowState(user_task="Add a feature")
    state.transition_to(WorkflowStatus.PLANNING)
    state.transition_to(WorkflowStatus.INSPECTING)
    state.transition_to(WorkflowStatus.PROPOSING_CHANGE)
    state.transition_to(WorkflowStatus.WAITING_FOR_APPROVAL)
    state.transition_to(WorkflowStatus.IMPLEMENTING)
    state.transition_to(WorkflowStatus.TESTING)
    state.transition_to(WorkflowStatus.ANALYZING)
    state.transition_to(WorkflowStatus.FIXING)
    state.transition_to(WorkflowStatus.PROPOSING_CHANGE)
    state.transition_to(WorkflowStatus.WAITING_FOR_APPROVAL)
    state.transition_to(WorkflowStatus.IMPLEMENTING)
    state.transition_to(WorkflowStatus.RETESTING)
    state.transition_to(WorkflowStatus.REGRESSION_TESTING)
    state.transition_to(WorkflowStatus.REVIEWING)
    state.transition_to(WorkflowStatus.COMPLETED)
    assert state.status == WorkflowStatus.COMPLETED


def test_mark_step_complete_moves_step_between_lists():
    state = WorkflowState(user_task="Add a feature")
    state.pending_steps = ["Inspect project", "Run tests"]
    state.mark_step_complete("Inspect project")
    assert state.pending_steps == ["Run tests"]
    assert state.completed_steps == ["Inspect project"]


def test_can_retry_respects_max_repair_attempts():
    state = WorkflowState(user_task="Fix a bug")
    assert MAX_REPAIR_ATTEMPTS == 3
    state.retry_count = 0
    assert state.can_retry() is True
    state.retry_count = MAX_REPAIR_ATTEMPTS
    assert state.can_retry() is False


def test_workflow_state_never_stores_full_prose_by_default():
    # errors/tool_results are meant to hold short one-line summaries, not
    # essays - this doesn't enforce a hard limit, but confirms the default
    # containers start empty (no accidental prefilled chain-of-thought).
    state = WorkflowState(user_task="Add a feature")
    assert state.tool_results == []
    assert state.errors == []
    assert state.task_plan == []


# ---------------------------------------------------------------------------
# Pending-change registry (human approval gate)
# ---------------------------------------------------------------------------


def test_register_change_is_not_approved_by_default():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert change.approved is False
    assert change.applied is False


def test_approve_change_sets_approved_flag():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert approve_change(change.change_id) is True
    assert get_change(change.change_id).approved is True


def test_approve_change_returns_false_for_unknown_id():
    assert approve_change("does-not-exist") is False


def test_reject_change_removes_it_from_registry():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert reject_change(change.change_id) is True
    assert get_change(change.change_id) is None


def test_reject_change_returns_false_for_unknown_id():
    assert reject_change("does-not-exist") is False


def test_list_pending_changes_excludes_applied():
    change1 = register_change("a.py", "create", "a = 1\n", "demo a")
    change2 = register_change("b.py", "create", "b = 2\n", "demo b")
    mark_applied(change1.change_id)
    pending = list_pending_changes()
    assert change1 not in pending
    assert change2 in pending


def test_clear_all_changes_empties_registry():
    register_change("a.py", "create", "a = 1\n", "demo")
    clear_all_changes()
    assert list_pending_changes() == []


# ---------------------------------------------------------------------------
# Repair-attempt circuit breaker
# ---------------------------------------------------------------------------


def test_repair_attempts_start_at_zero():
    assert repair_attempts_for("a.py") == 0
    assert repair_limit_reached("a.py") is False


def test_record_apply_increments_per_file():
    record_apply("a.py")
    record_apply("a.py")
    assert repair_attempts_for("a.py") == 2
    assert repair_attempts_for("b.py") == 0


def test_repair_limit_reached_at_max_attempts():
    for _ in range(MAX_REPAIR_ATTEMPTS):
        record_apply("a.py")
    assert repair_limit_reached("a.py") is True


def test_repair_limit_not_reached_below_max_attempts():
    for _ in range(MAX_REPAIR_ATTEMPTS - 1):
        record_apply("a.py")
    assert repair_limit_reached("a.py") is False


def test_record_test_outcome_passed_clears_all_files():
    record_apply("a.py")
    record_apply("b.py")
    record_test_outcome(passed=True)
    assert repair_attempts_for("a.py") == 0
    assert repair_attempts_for("b.py") == 0


def test_record_test_outcome_failed_does_not_clear():
    record_apply("a.py")
    record_test_outcome(passed=False)
    assert repair_attempts_for("a.py") == 1


def test_reset_repair_attempts_for_one_file():
    record_apply("a.py")
    record_apply("b.py")
    reset_repair_attempts("a.py")
    assert repair_attempts_for("a.py") == 0
    assert repair_attempts_for("b.py") == 1


def test_reset_repair_attempts_for_all_files():
    record_apply("a.py")
    record_apply("b.py")
    reset_repair_attempts()
    assert repair_attempts_for("a.py") == 0
    assert repair_attempts_for("b.py") == 0


def test_repair_attempts_snapshot_reflects_current_state():
    record_apply("a.py")
    record_apply("a.py")
    record_apply("b.py")
    snapshot = repair_attempts_snapshot()
    assert snapshot == {"a.py": 2, "b.py": 1}
