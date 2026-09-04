"""
Tests for tools.py's Phase 6 controlled file-change tools:
propose_file_change and apply_approved_change.

These exercise the real tools directly (via .invoke(...)) against this
project's actual filesystem, the same way the other tool test files do -
any file created during a test is cleaned up afterward.
"""

import pytest

import workflow
from tools import (
    PROJECT_ROOT,
    apply_approved_change,
    list_pending_changes,
    propose_file_change,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()
    yield
    workflow.clear_all_changes()
    workflow.reset_repair_attempts()


@pytest.fixture
def temp_project_file():
    """A safe, disposable path inside the project root for create/modify
    tests - removed after the test regardless of outcome."""
    rel_path = "tests/_phase6_scratch_file.py"
    abs_path = PROJECT_ROOT / rel_path
    yield rel_path
    if abs_path.exists():
        abs_path.unlink()


def _extract_change_id(propose_result: str) -> str:
    # "Proposed change registered: id=<hex>, action=..."
    marker = "id="
    start = propose_result.index(marker) + len(marker)
    end = propose_result.index(",", start)
    return propose_result[start:end]


# ---------------------------------------------------------------------------
# propose_file_change: never writes, validates path safety
# ---------------------------------------------------------------------------


def test_propose_file_change_does_not_write_to_disk(temp_project_file):
    result = propose_file_change.invoke(
        {
            "file_path": temp_project_file,
            "new_content": "x = 1\n",
            "reason": "test",
        }
    )
    assert "not been written to disk" in result.lower()
    assert not (PROJECT_ROOT / temp_project_file).exists()


def test_propose_file_change_registers_a_pending_change(temp_project_file):
    result = propose_file_change.invoke(
        {
            "file_path": temp_project_file,
            "new_content": "x = 1\n",
            "reason": "test",
        }
    )
    change_id = _extract_change_id(result)
    change = workflow.get_change(change_id)
    assert change is not None
    assert change.approved is False
    assert change.file_path == temp_project_file


def test_propose_file_change_detects_create_vs_modify(temp_project_file):
    create_result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    assert "action=create" in create_result

    modify_result = propose_file_change.invoke(
        {"file_path": "tools.py", "new_content": "# unused\n", "reason": "t"}
    )
    assert "action=modify" in modify_result
    # tools.py itself must never actually be touched by this test.
    change_id = _extract_change_id(modify_result)
    workflow.reject_change(change_id)


def test_propose_file_change_flags_suspicious_truncation_as_high_risk():
    """Regression test for a real failure observed in live testing: the
    model proposed replacing tools.py (1600+ lines) with only its first
    ~300 lines plus a placeholder comment ("... rest of tools.py omitted for
    brevity"), which would have deleted most of the project's tools if a
    human had approved it without noticing. A "modify" proposal that is
    drastically shorter than the file it replaces must be flagged high risk
    with an explicit warning, since a human approver may not otherwise
    notice a silent truncation."""
    modify_result = propose_file_change.invoke(
        {
            "file_path": "tools.py",
            "new_content": "# rest of tools.py omitted for brevity\n",
            "reason": "t",
        }
    )
    assert "risk=high" in modify_result
    assert "warning" in modify_result.lower()
    assert "shorter than the current file" in modify_result.lower()

    change_id = _extract_change_id(modify_result)
    change = workflow.get_change(change_id)
    assert change.risk == "high"
    workflow.reject_change(change_id)


def test_propose_file_change_does_not_flag_normal_sized_modification():
    current_length = len((PROJECT_ROOT / "tools.py").read_text(encoding="utf-8"))
    # Comparable size to the real file (90%) - a normal edit, not a truncation.
    modify_result = propose_file_change.invoke(
        {
            "file_path": "tools.py",
            "new_content": "x = 1\n" * int(current_length * 0.9 / len("x = 1\n")),
            "reason": "t",
        }
    )
    assert "risk=medium" in modify_result
    assert "warning" not in modify_result.lower()

    change_id = _extract_change_id(modify_result)
    workflow.reject_change(change_id)


def test_propose_file_change_new_file_creation_is_never_flagged_as_truncation(
    temp_project_file,
):
    # A brand-new small file is a "create", not a "modify" - it has nothing
    # to be truncated relative to, so it must never trigger the warning.
    result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    assert "risk=high" not in result
    assert "warning" not in result.lower()


def test_propose_file_change_rejects_path_traversal():
    result = propose_file_change.invoke(
        {"file_path": "../../evil.py", "new_content": "x = 1", "reason": "bad"}
    )
    assert "error" in result.lower()
    assert "outside the project" in result.lower()


def test_propose_file_change_rejects_env_file():
    result = propose_file_change.invoke(
        {"file_path": ".env", "new_content": "SECRET=1", "reason": "bad"}
    )
    assert "error" in result.lower()
    assert "security" in result.lower()


def test_propose_file_change_rejects_empty_path():
    result = propose_file_change.invoke(
        {"file_path": "", "new_content": "x = 1", "reason": "bad"}
    )
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# apply_approved_change: hard-gated on workflow.approve_change()
# ---------------------------------------------------------------------------


def test_apply_approved_change_refuses_unapproved_change(temp_project_file):
    propose_result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    change_id = _extract_change_id(propose_result)

    apply_result = apply_approved_change.invoke({"change_id": change_id})

    assert "not been approved" in apply_result.lower()
    assert not (PROJECT_ROOT / temp_project_file).exists()


def test_apply_approved_change_writes_file_once_approved(temp_project_file):
    propose_result = propose_file_change.invoke(
        {
            "file_path": temp_project_file,
            "new_content": "VALUE = 42\n",
            "reason": "t",
        }
    )
    change_id = _extract_change_id(propose_result)

    workflow.approve_change(change_id)
    apply_result = apply_approved_change.invoke({"change_id": change_id})

    assert "applied change" in apply_result.lower()
    written_path = PROJECT_ROOT / temp_project_file
    assert written_path.exists()
    assert written_path.read_text(encoding="utf-8") == "VALUE = 42\n"


def test_apply_approved_change_unknown_id_is_an_error():
    result = apply_approved_change.invoke({"change_id": "not-a-real-id"})
    assert "no pending change found" in result.lower()


def test_apply_approved_change_cannot_be_tricked_by_approving_a_different_id(
    temp_project_file,
):
    propose_result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    real_change_id = _extract_change_id(propose_result)

    # Approving some unrelated id must not approve the real pending change.
    workflow.approve_change("unrelated-id-1234")
    apply_result = apply_approved_change.invoke({"change_id": real_change_id})

    assert "not been approved" in apply_result.lower()
    assert not (PROJECT_ROOT / temp_project_file).exists()


def test_apply_approved_change_still_enforces_path_safety_at_apply_time():
    # Even if a change were somehow registered for a blocked path, apply
    # must independently re-check path safety rather than trusting the
    # registry blindly. Deliberately never reads .env's real content here -
    # only the tool's own refusal message is asserted, so no secret value
    # can ever leak into test output.
    change = workflow.register_change(
        file_path=".env", action="modify", content="X=1", reason="bad"
    )
    workflow.approve_change(change.change_id)
    result = apply_approved_change.invoke({"change_id": change.change_id})
    assert "security" in result.lower()


def test_full_propose_approve_apply_cycle_matches_registry_state(temp_project_file):
    propose_result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "READY = True\n", "reason": "t"}
    )
    change_id = _extract_change_id(propose_result)
    assert workflow.get_change(change_id).applied is False

    workflow.approve_change(change_id)
    apply_approved_change.invoke({"change_id": change_id})

    assert workflow.get_change(change_id).applied is True
    assert change_id not in [c.change_id for c in workflow.list_pending_changes()]


# ---------------------------------------------------------------------------
# apply_approved_change: real, code-level repair-attempt circuit breaker
# ---------------------------------------------------------------------------


def _propose_approve_apply(file_path: str, content: str) -> str:
    propose_result = propose_file_change.invoke(
        {"file_path": file_path, "new_content": content, "reason": "t"}
    )
    change_id = _extract_change_id(propose_result)
    workflow.approve_change(change_id)
    return apply_approved_change.invoke({"change_id": change_id})


def test_apply_approved_change_enforces_repair_attempt_limit(temp_project_file):
    """Regression test: MAX_REPAIR_ATTEMPTS must be a real, code-level limit
    on apply_approved_change, not just a system-prompt instruction the model
    might ignore. After MAX_REPAIR_ATTEMPTS applies to the same file with no
    passing run_pytest in between, further applies must be refused."""
    for i in range(workflow.MAX_REPAIR_ATTEMPTS):
        result = _propose_approve_apply(temp_project_file, f"attempt = {i}\n")
        assert "applied change" in result.lower(), f"attempt {i} should succeed"

    # One more, still with no passing test run in between - must be refused.
    result = _propose_approve_apply(temp_project_file, "attempt = final\n")
    assert "repair-attempt limit" in result.lower() or "limit has been reached" in (
        result.lower()
    )
    # The file must still hold the last successfully applied content, not
    # the refused one.
    written = (PROJECT_ROOT / temp_project_file).read_text(encoding="utf-8")
    assert written == f"attempt = {workflow.MAX_REPAIR_ATTEMPTS - 1}\n"


def test_apply_approved_change_limit_resets_after_passing_test(temp_project_file):
    for i in range(workflow.MAX_REPAIR_ATTEMPTS):
        _propose_approve_apply(temp_project_file, f"attempt = {i}\n")

    workflow.record_test_outcome(passed=True)  # simulates a real passing run_pytest

    result = _propose_approve_apply(temp_project_file, "attempt = after_reset\n")
    assert "applied change" in result.lower()


def test_apply_approved_change_limit_is_scoped_per_file(temp_project_file):
    other_file = "tests/_phase6_scratch_file_other.py"
    try:
        for i in range(workflow.MAX_REPAIR_ATTEMPTS):
            _propose_approve_apply(temp_project_file, f"attempt = {i}\n")

        # A different file must be unaffected by another file's stuck loop.
        result = _propose_approve_apply(other_file, "x = 1\n")
        assert "applied change" in result.lower()
    finally:
        other_path = PROJECT_ROOT / other_file
        if other_path.exists():
            other_path.unlink()


# ---------------------------------------------------------------------------
# list_pending_changes
# ---------------------------------------------------------------------------


def test_list_pending_changes_reports_no_pending_changes_when_empty():
    result = list_pending_changes.invoke({})
    assert "no pending changes" in result.lower()


def test_list_pending_changes_shows_real_change_details(temp_project_file):
    propose_result = propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    change_id = _extract_change_id(propose_result)

    result = list_pending_changes.invoke({})
    assert change_id in result
    assert temp_project_file in result
    assert "approved=False" in result

    workflow.approve_change(change_id)
    result = list_pending_changes.invoke({})
    assert "approved=True" in result


def test_list_pending_changes_never_writes_anything(temp_project_file):
    propose_file_change.invoke(
        {"file_path": temp_project_file, "new_content": "x = 1\n", "reason": "t"}
    )
    list_pending_changes.invoke({})
    assert not (PROJECT_ROOT / temp_project_file).exists()
