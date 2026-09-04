"""
workflow.py
-----------
Phase 6: structured, observable state for controlled autonomous development
tasks (plan -> inspect -> implement -> test -> review), plus the human
approval gate that controlled file changes must pass through before they are
ever written to disk.

This module holds only plain data and in-memory registries - it has no
filesystem, LangChain, or Streamlit access of its own:
    - WorkflowStatus / WorkflowState track a task's concise operational
      progress (plan, completed/pending steps, files touched, test results,
      retry count) for display in the UI/logs. Never chain-of-thought, never
      secrets.
    - ProposedChange + the pending-change registry are the human-approval
      gate for tools.py's propose_file_change/apply_approved_change: a
      change can only move from "proposed" to "written to disk" once
      approve_change() has been called, which only app.py's UI (or cli.py's
      approval prompt) ever calls - the AI agent itself has no tool that can
      approve its own change.

Single-user simplification: the pending-change registry is a plain
module-level dict, not per-session/per-user storage. This project runs as a
local, single-user Streamlit app or CLI (see app.py/cli.py) rather than a
multi-tenant service, so one shared in-process registry is intentional, not
an oversight - it would need to become per-session state before this project
was ever served to multiple concurrent users.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum

MAX_REPAIR_ATTEMPTS = 3


class WorkflowStatus(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    INSPECTING = "INSPECTING"
    SEARCHING_DOCUMENTATION = "SEARCHING_DOCUMENTATION"
    PROPOSING_CHANGE = "PROPOSING_CHANGE"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    ANALYZING = "ANALYZING"
    FIXING = "FIXING"
    RETESTING = "RETESTING"
    REGRESSION_TESTING = "REGRESSION_TESTING"
    REVIEWING = "REVIEWING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Controlled state transitions - keeps the workflow from jumping into a
# nonsensical state or looping forever. A transition not listed here (and
# not a same-state no-op) raises ValueError in WorkflowState.transition_to().
_ALLOWED_TRANSITIONS: dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.IDLE: {WorkflowStatus.PLANNING},
    WorkflowStatus.PLANNING: {WorkflowStatus.INSPECTING, WorkflowStatus.FAILED},
    WorkflowStatus.INSPECTING: {
        WorkflowStatus.SEARCHING_DOCUMENTATION,
        WorkflowStatus.PROPOSING_CHANGE,
        WorkflowStatus.IMPLEMENTING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.SEARCHING_DOCUMENTATION: {
        WorkflowStatus.PROPOSING_CHANGE,
        WorkflowStatus.IMPLEMENTING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.PROPOSING_CHANGE: {
        WorkflowStatus.WAITING_FOR_APPROVAL,
        WorkflowStatus.IMPLEMENTING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.IMPLEMENTING: {
        WorkflowStatus.WAITING_FOR_APPROVAL,
        WorkflowStatus.TESTING,
        WorkflowStatus.RETESTING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.WAITING_FOR_APPROVAL: {
        WorkflowStatus.IMPLEMENTING,
        WorkflowStatus.TESTING,
        WorkflowStatus.PROPOSING_CHANGE,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.TESTING: {
        WorkflowStatus.ANALYZING,
        WorkflowStatus.REGRESSION_TESTING,
        WorkflowStatus.REVIEWING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.ANALYZING: {WorkflowStatus.FIXING, WorkflowStatus.FAILED},
    WorkflowStatus.FIXING: {
        WorkflowStatus.RETESTING,
        WorkflowStatus.PROPOSING_CHANGE,
        WorkflowStatus.WAITING_FOR_APPROVAL,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.RETESTING: {
        WorkflowStatus.REGRESSION_TESTING,
        WorkflowStatus.REVIEWING,
        WorkflowStatus.ANALYZING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.REGRESSION_TESTING: {
        WorkflowStatus.REVIEWING,
        WorkflowStatus.ANALYZING,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.REVIEWING: {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED},
    WorkflowStatus.COMPLETED: set(),
    WorkflowStatus.FAILED: set(),
}


@dataclass
class ProposedChange:
    """One human-reviewable file create/modify proposal. `content` is the
    file's full proposed new text - never executed, only ever written
    verbatim to `file_path` by apply_approved_change(), and only after
    `approved` is True."""

    change_id: str
    file_path: str
    action: str  # "create" or "modify"
    content: str
    reason: str
    risk: str = "medium"  # "low" | "medium" | "high"
    approved: bool = False
    applied: bool = False


@dataclass
class WorkflowState:
    """Concise, operational tracking for one autonomous development task.

    Deliberately holds only short summaries (task_plan steps, tool-result
    one-liners, error messages) - never full chain-of-thought and never
    secrets. `tool_results`/`errors` entries should already be the
    human-readable summary a UI can show directly.
    """

    user_task: str
    status: WorkflowStatus = WorkflowStatus.IDLE
    task_plan: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)
    proposed_change_ids: list[str] = field(default_factory=list)
    test_results: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    retry_count: int = 0
    final_status: str | None = None

    def transition_to(self, new_status: WorkflowStatus) -> None:
        """Move to `new_status`, enforcing _ALLOWED_TRANSITIONS.

        Raises ValueError for a transition that isn't allowed, so a caller
        can never silently push the workflow into an invalid or looping
        state.
        """
        if new_status == self.status:
            return
        allowed = _ALLOWED_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid workflow transition: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status

    def mark_step_complete(self, step: str) -> None:
        if step in self.pending_steps:
            self.pending_steps.remove(step)
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def can_retry(self) -> bool:
        """True if another repair attempt is still allowed under
        MAX_REPAIR_ATTEMPTS - the hard ceiling that keeps the
        test -> analyze -> fix -> retest loop from running forever."""
        return self.retry_count < MAX_REPAIR_ATTEMPTS


# ---------------------------------------------------------------------------
# Pending-change registry: the human-approval gate. See module docstring for
# why this is a process-local, single-user store rather than per-session.
# ---------------------------------------------------------------------------

_pending_changes: dict[str, ProposedChange] = {}


def register_change(
    file_path: str, action: str, content: str, reason: str, risk: str = "medium"
) -> ProposedChange:
    """Register a new proposed change and return it. Never writes to disk -
    callers (tools.py's propose_file_change) are responsible for path
    safety checks before calling this."""
    change_id = uuid.uuid4().hex[:8]
    change = ProposedChange(
        change_id=change_id,
        file_path=file_path,
        action=action,
        content=content,
        reason=reason,
        risk=risk,
    )
    _pending_changes[change_id] = change
    return change


def get_change(change_id: str) -> ProposedChange | None:
    return _pending_changes.get(change_id)


def approve_change(change_id: str) -> bool:
    """Mark a change approved. Only ever called from the human-facing UI
    (app.py's Approve button) or CLI approval prompt - never exposed as a
    tool the AI agent itself can call, so the agent can never approve its
    own proposed change."""
    change = _pending_changes.get(change_id)
    if change is None:
        return False
    change.approved = True
    return True


def reject_change(change_id: str) -> bool:
    """Discard a pending change without ever writing it."""
    return _pending_changes.pop(change_id, None) is not None


def list_pending_changes() -> list[ProposedChange]:
    """Proposed changes that have not yet been written to disk (approved or not)."""
    return [c for c in _pending_changes.values() if not c.applied]


def mark_applied(change_id: str) -> None:
    change = _pending_changes.get(change_id)
    if change is not None:
        change.applied = True


def clear_all_changes() -> None:
    """Discard every pending change without applying any of them."""
    _pending_changes.clear()


# ---------------------------------------------------------------------------
# Repair-loop circuit breaker: a REAL code-level enforcement of
# MAX_REPAIR_ATTEMPTS, not just a system-prompt instruction the model
# follows. Scoped per file rather than globally, so unrelated tasks never
# interfere with each other: a "repair loop" is fundamentally about
# repeatedly modifying and testing ONE file until it works, so tallying
# per file_path is both the natural unit and avoids one file's stuck loop
# from ever blocking unrelated work on a different file.
# ---------------------------------------------------------------------------

_repair_attempts_by_file: dict[str, int] = {}


def record_apply(file_path: str) -> None:
    """Call this once a file change has actually been written to disk by
    apply_approved_change. Counts toward that file's repair-attempt tally
    until a full passing test run (record_test_outcome(True)) clears it."""
    _repair_attempts_by_file[file_path] = _repair_attempts_by_file.get(file_path, 0) + 1


def record_test_outcome(passed: bool) -> None:
    """Call this every time run_pytest actually runs. A passing run means
    the suite is green again, so every file's repair-attempt tally is
    cleared - whatever was being repaired is now considered resolved. A
    failing run intentionally does nothing here; attribution of *which*
    file caused the failure happens via record_apply, called separately
    each time that file is actually re-applied."""
    if passed:
        _repair_attempts_by_file.clear()


def repair_attempts_for(file_path: str) -> int:
    return _repair_attempts_by_file.get(file_path, 0)


def repair_attempts_snapshot() -> dict[str, int]:
    """A read-only copy of every file's current repair-attempt tally, for
    the UI to show which files (if any) have hit the limit."""
    return dict(_repair_attempts_by_file)


def repair_limit_reached(file_path: str) -> bool:
    """True once `file_path` has been applied MAX_REPAIR_ATTEMPTS times
    without an intervening passing test run - the hard ceiling that stops
    apply_approved_change from ever being used to run an unlimited
    propose -> apply -> test repair loop on the same file."""
    return repair_attempts_for(file_path) >= MAX_REPAIR_ATTEMPTS


def reset_repair_attempts(file_path: str | None = None) -> None:
    """Manually clear the repair-attempt tally - for a specific file, or
    for every file if `file_path` is omitted. Only ever called from the
    human-facing UI (a "Reset repair counter" control), the same way
    approve_change/reject_change are - lets a human deliberately allow more
    attempts on a file that genuinely needs them, rather than the agent
    ever being able to lift its own limit."""
    if file_path is None:
        _repair_attempts_by_file.clear()
    else:
        _repair_attempts_by_file.pop(file_path, None)
