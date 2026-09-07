import os
import pytest
from database import init_db, get_connection
from department import add_department, get_departments, update_department, delete_department
from employee import add_employee, get_employees, update_employee, delete_employee, validate_email

@pytest.fixture(autouse=True)
def setup_test_db():
    if os.path.exists("test_ems.db"):
        os.remove("test_ems.db")
    init_db("test_ems.db")
    yield
    if os.path.exists("test_ems.db"):
        os.remove("test_ems.db")

def test_database_initialization():
    conn = get_connection("test_ems.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "departments" in tables
    assert "employees" in tables

def test_department_crud():
    # Add department
    success, msg = add_department("Engineering", "test_ems.db")
    assert success
    
    depts = get_departments("test_ems.db")
    assert len(depts) == 1
    assert depts[0]["name"] == "Engineering"
    
    # Duplicate department
    success, msg = add_department("Engineering", "test_ems.db")
    assert not success
    
    # Update department
    success, msg = update_department(depts[0]["id"], "R&D", "test_ems.db")
    assert success
    depts = get_departments("test_ems.db")
    assert depts[0]["name"] == "R&D"

def test_employee_validation_and_creation():
    add_department("HR", "test_ems.db")
    
    # Valid employee
    emp_data = {
        "employee_id": "EMP001",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "phone": "555-0100",
        "department": "HR",
        "designation": "Manager",
        "salary": 75000.0,
        "joining_date": "2022-05-01"
    }
    success, msg = add_employee(emp_data, "test_ems.db")
    assert success
    
    # Duplicate ID
    success, msg = add_employee(emp_data, "test_ems.db")
    assert not success
    
    # Invalid email
    emp_data2 = emp_data.copy()
    emp_data2["employee_id"] = "EMP002"
    emp_data2["email"] = "invalid-email"
    success, msg = add_employee(emp_data2, "test_ems.db")
    assert not success
    assert "email" in msg.lower()
    
    # Negative salary
    emp_data3 = emp_data.copy()
    emp_data3["employee_id"] = "EMP003"
    emp_data3["salary"] = -1000.0
    success, msg = add_employee(emp_data3, "test_ems.db")
    assert not success
    assert "salary" in msg.lower()

def test_employee_search_and_filter():
    add_department("Sales", "test_ems.db")
    add_department("IT", "test_ems.db")
    
    add_employee({
        "employee_id": "S01",
        "name": "Bob Jones",
        "email": "bob@sales.com",
        "phone": "123",
        "department": "Sales",
        "designation": "Rep",
        "salary": 50000,
        "joining_date": "2023-01-01"
    }, "test_ems.db")
    
    add_employee({
        "employee_id": "I01",
        "name": "Charlie Brown",
        "email": "charlie@it.com",
        "phone": "456",
        "department": "IT",
        "designation": "Dev",
        "salary": 80000,
        "joining_date": "2023-02-01"
    }, "test_ems.db")
    
    # Search by name
    res = get_employees(search_query="Bob", db_path="test_ems.db")
    assert len(res) == 1
    assert res[0]["employee_id"] == "S01"
    
    # Filter by department
    res_it = get_employees(dept_filter="IT", db_path="test_ems.db")
    assert len(res_it) == 1
    assert res_it[0]["name"] == "Charlie Brown"

def test_employee_editing_and_deletion():
    add_department("Admin", "test_ems.db")
    add_employee({
        "employee_id": "A01",
        "name": "David",
        "email": "david@admin.com",
        "phone": "789",
        "department": "Admin",
        "designation": "Clerk",
        "salary": 40000,
        "joining_date": "2023-03-01"
    }, "test_ems.db")
    
    # Edit
    updated_data = {
        "name": "David Miller",
        "email": "david.miller@admin.com",
        "phone": "789",
        "department": "Admin",
        "designation": "Senior Clerk",
        "salary": 45000,
        "joining_date": "2023-03-01"
    }
    success, msg = update_employee("A01", updated_data, "test_ems.db")
    assert success
    
    emps = get_employees(search_query="A01", db_path="test_ems.db")
    assert emps[0]["name"] == "David Miller"
    assert emps[0]["salary"] == 45000
    
    # Delete
    success, msg = delete_employee("A01", "test_ems.db")
    assert success
    assert len(get_employees(db_path="test_ems.db")) == 0

def test_department_deletion_restriction():
    add_department("HR", "test_ems.db")
    depts = get_departments("test_ems.db")
    dept_id = depts[0]["id"]
    
    add_employee({
        "employee_id": "HR01",
        "name": "Eve",
        "email": "eve@hr.com",
        "phone": "111",
        "department": "HR",
        "designation": "HR",
        "salary": 60000,
        "joining_date": "2023-01-01"
    }, "test_ems.db")
    
    # Try deleting department with assigned employee
    success, msg = delete_department(dept_id, "test_ems.db")
    assert not success
    assert "assigned employee" in msg.lower()
    
    # Delete employee first
    delete_employee("HR01", "test_ems.db")
    success, msg = delete_department(dept_id, "test_ems.db")
    assert success
