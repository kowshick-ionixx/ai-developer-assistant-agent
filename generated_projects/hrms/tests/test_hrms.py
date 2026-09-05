import os
import pytest
from database import (
    init_db,
    add_employee,
    get_employees,
    get_employee_by_id,
    mark_attendance,
    get_attendance,
    apply_leave,
    get_leaves,
    update_leave_status
)

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_hrms.db"
    db_path = str(db_file)
    init_db(db_path)
    return db_path

def test_add_and_get_employee(test_db):
    success, msg = add_employee("Alice Smith", "alice@example.com", "Engineering", "Developer", "2023-01-01", 75000.0, db_path=test_db)
    assert success is True
    
    # Duplicate email should fail
    success_dup, _ = add_employee("Alice Clone", "alice@example.com", "HR", "Manager", "2023-02-01", 80000.0, db_path=test_db)
    assert success_dup is False
    
    employees = get_employees(db_path=test_db)
    assert len(employees) == 1
    assert employees[0]["name"] == "Alice Smith"
    
    emp = get_employee_by_id(employees[0]["id"], db_path=test_db)
    assert emp is not None
    assert emp["email"] == "alice@example.com"

def test_attendance(test_db):
    add_employee("Bob Jones", "bob@example.com", "Sales", "Rep", "2023-01-01", 50000.0, db_path=test_db)
    employees = get_employees(db_path=test_db)
    emp_id = employees[0]["id"]
    
    mark_attendance(emp_id, "2023-10-01", "Present", db_path=test_db)
    records = get_attendance(db_path=test_db)
    assert len(records) == 1
    assert records[0]["status"] == "Present"
    
    # Update attendance
    mark_attendance(emp_id, "2023-10-01", "Late", db_path=test_db)
    records = get_attendance(db_path=test_db)
    assert len(records) == 1
    assert records[0]["status"] == "Late"

def test_leave_management(test_db):
    add_employee("Charlie Brown", "charlie@example.com", "Finance", "Analyst", "2023-01-01", 60000.0, db_path=test_db)
    employees = get_employees(db_path=test_db)
    emp_id = employees[0]["id"]
    
    apply_leave(emp_id, "2023-11-01", "2023-11-03", "Vacation", db_path=test_db)
    leaves = get_leaves(db_path=test_db)
    assert len(leaves) == 1
    assert leaves[0]["status"] == "Pending"
    
    update_leave_status(leaves[0]["id"], "Approved", db_path=test_db)
    leaves = get_leaves(db_path=test_db)
    assert leaves[0]["status"] == "Approved"
