import pytest
from database import (
    add_employee,
    apply_leave,
    get_all_employees,
    get_attendance,
    get_leaves,
    init_db,
    mark_attendance,
    update_leave_status,
)


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test_hrms.db")
    init_db(db_path)
    return db_path


def test_add_and_get_employee(temp_db):
    success, msg = add_employee(
        "Alice Smith",
        "alice@example.com",
        "Engineering",
        "Developer",
        "2023-01-01",
        db_path=temp_db,
    )
    assert success is True

    # Duplicate email check
    success_dup, _ = add_employee(
        "Alice Duplicate",
        "alice@example.com",
        "HR",
        "Manager",
        "2023-02-01",
        db_path=temp_db,
    )
    assert success_dup is False

    employees = get_all_employees(db_path=temp_db)
    assert len(employees) == 1
    assert employees[0]["name"] == "Alice Smith"
    assert employees[0]["email"] == "alice@example.com"


def test_attendance(temp_db):
    add_employee("Bob Jones", "bob@example.com", "Sales", "Executive", db_path=temp_db)
    employees = get_all_employees(db_path=temp_db)
    emp_id = employees[0]["id"]

    success, msg = mark_attendance(emp_id, "2023-10-01", "Present", db_path=temp_db)
    assert success is True

    logs = get_attendance(db_path=temp_db)
    assert len(logs) == 1
    assert logs[0]["name"] == "Bob Jones"
    assert logs[0]["status"] == "Present"


def test_leave_management(temp_db):
    add_employee(
        "Charlie Brown", "charlie@example.com", "Marketing", "Designer", db_path=temp_db
    )
    employees = get_all_employees(db_path=temp_db)
    emp_id = employees[0]["id"]

    success, msg = apply_leave(
        emp_id, "2023-11-01", "2023-11-05", "Vacation", db_path=temp_db
    )
    assert success is True

    leaves = get_leaves(db_path=temp_db)
    assert len(leaves) == 1
    assert leaves[0]["status"] == "Pending"

    leave_id = leaves[0]["id"]
    update_leave_status(leave_id, "Approved", db_path=temp_db)

    updated_leaves = get_leaves(db_path=temp_db)
    assert updated_leaves[0]["status"] == "Approved"
