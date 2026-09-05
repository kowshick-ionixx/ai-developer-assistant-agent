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
    approve_change_set,
    clear_all_changes,
    get_change,
    get_current_changeset_id,
    list_changeset,
    list_pending_changes,
    mark_applied,
    record_apply,
    record_test_outcome,
    register_change,
    reject_change,
    reject_change_set,
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


def test_approving_an_already_approved_change_is_a_safe_noop():
    """Regression test for the duplicate-approval bug: a second approval
    attempt on the same change_id (e.g. a duplicate click, or a Streamlit
    rerun re-delivering the same click) must be ignored - it must not
    re-announce or re-process the approval - while the change stays
    approved exactly once."""
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert approve_change(change.change_id) is True
    assert approve_change(change.change_id) is False
    assert approve_change(change.change_id) is False
    assert get_change(change.change_id).approved is True


def test_approving_an_already_applied_change_is_a_safe_noop():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    approve_change(change.change_id)
    mark_applied(change.change_id)
    assert approve_change(change.change_id) is False
    assert get_change(change.change_id).applied is True


def test_approving_two_different_changes_each_transitions_exactly_once():
    change_a = register_change("a.py", "create", "a = 1\n", "demo a")
    change_b = register_change("b.py", "create", "b = 2\n", "demo b")
    assert approve_change(change_a.change_id) is True
    assert approve_change(change_b.change_id) is True
    assert approve_change(change_a.change_id) is False
    assert approve_change(change_b.change_id) is False
    assert get_change(change_a.change_id).approved is True
    assert get_change(change_b.change_id).approved is True


def test_unapproved_change_cannot_be_applied():
    """apply_approved_change (tools.py) refuses anything not approved; this
    confirms the registry-level invariant it relies on: a fresh change is
    never approved by default, so it is never eligible to be treated as
    applied without going through approve_change first."""
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert change.approved is False
    assert change.applied is False


def test_reject_change_removes_it_from_registry():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert reject_change(change.change_id) is True
    assert get_change(change.change_id) is None


def test_reject_change_returns_false_for_unknown_id():
    assert reject_change("does-not-exist") is False


def test_rejected_change_can_never_be_approved_or_applied():
    """A rejected change is discarded outright, so it can never be
    resurrected into an approved (and therefore applicable) state by a
    later approve_change call for the same id."""
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    change_id = change.change_id
    assert reject_change(change_id) is True
    assert approve_change(change_id) is False
    assert get_change(change_id) is None


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
# Change sets: every file proposed for one development task is grouped and
# approved/rejected together as a single unit - never one at a time.
# ---------------------------------------------------------------------------


def test_a_single_proposed_change_gets_its_own_changeset():
    change = register_change("foo.py", "create", "x = 1\n", "demo")
    assert change.changeset_id != ""
    assert get_current_changeset_id() == change.changeset_id
    assert list_changeset(change.changeset_id) == [change]


def test_multiple_changes_proposed_before_approval_share_one_changeset():
    change_a = register_change("a.py", "create", "a = 1\n", "demo a")
    change_b = register_change("b.py", "create", "b = 2\n", "demo b")
    assert change_a.changeset_id == change_b.changeset_id
    assert get_current_changeset_id() == change_a.changeset_id
    assert list_changeset(change_a.changeset_id) == [change_a, change_b]


def test_a_new_changeset_starts_only_once_the_previous_one_is_fully_applied():
    """A later change (e.g. a repair-loop fix proposed after a failing test
    run) must NOT join an earlier changeset whose members are all already
    applied - it needs its own fresh approval cycle."""
    change_a = register_change("a.py", "create", "a = 1\n", "demo a")
    mark_applied(change_a.change_id)
    change_b = register_change("b.py", "create", "b = 2\n", "demo b")
    assert change_b.changeset_id != change_a.changeset_id
    assert get_current_changeset_id() == change_b.changeset_id


def test_get_current_changeset_id_is_none_when_nothing_is_pending():
    assert get_current_changeset_id() is None
    change = register_change("a.py", "create", "a = 1\n", "demo")
    mark_applied(change.change_id)
    assert get_current_changeset_id() is None


def test_approve_change_set_approves_every_member_at_once():
    change_a = register_change("a.py", "create", "a = 1\n", "demo a")
    change_b = register_change("b.py", "create", "b = 2\n", "demo b")

    newly_approved = approve_change_set(change_a.changeset_id)

    assert set(newly_approved) == {change_a.change_id, change_b.change_id}
    assert get_change(change_a.change_id).approved is True
    assert get_change(change_b.change_id).approved is True
    # Approve never applies.
    assert get_change(change_a.change_id).applied is False
    assert get_change(change_b.change_id).applied is False


def test_approve_change_set_is_idempotent():
    """Repeated approval does nothing: calling approve_change_set again
    after the whole set is already approved must approve nothing new."""
    change = register_change("a.py", "create", "a = 1\n", "demo")
    assert approve_change_set(change.changeset_id) == [change.change_id]
    assert approve_change_set(change.changeset_id) == []
    assert approve_change_set(change.changeset_id) == []
    assert get_change(change.change_id).approved is True


def test_approve_change_set_unknown_id_approves_nothing():
    assert approve_change_set("does-not-exist") == []


def test_reject_change_set_discards_every_member():
    change_a = register_change("a.py", "create", "a = 1\n", "demo a")
    change_b = register_change("b.py", "create", "b = 2\n", "demo b")

    rejected = reject_change_set(change_a.changeset_id)

    assert set(rejected) == {change_a.change_id, change_b.change_id}
    assert get_change(change_a.change_id) is None
    assert get_change(change_b.change_id) is None
    assert get_current_changeset_id() is None


def test_reject_change_set_is_idempotent():
    change = register_change("a.py", "create", "a = 1\n", "demo")
    assert reject_change_set(change.changeset_id) == [change.change_id]
    assert reject_change_set(change.changeset_id) == []


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
