import re
import sqlite3

from database import get_connection


def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, email))


def add_employee(data, db_path="ems.db"):
    # Required fields check
    required = [
        "employee_id",
        "name",
        "email",
        "phone",
        "department",
        "designation",
        "salary",
        "joining_date",
    ]
    for field in required:
        if not data.get(field) or not str(data[field]).strip():
            return False, f"Field '{field}' cannot be empty."

    if not validate_email(data["email"]):
        return False, "Invalid email format."

    try:
        salary = float(data["salary"])
        if salary <= 0:
            return False, "Salary must be a positive number."
    except ValueError:
        return False, "Salary must be a valid number."

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO employees (employee_id, name, email, phone, department, designation, salary, joining_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(data["employee_id"]).strip(),
                str(data["name"]).strip(),
                str(data["email"]).strip(),
                str(data["phone"]).strip(),
                str(data["department"]).strip(),
                str(data["designation"]).strip(),
                salary,
                str(data["joining_date"]).strip(),
            ),
        )
        conn.commit()
        return True, "Employee added successfully."
    except sqlite3.IntegrityError:
        return False, "Employee ID already exists."
    finally:
        conn.close()


def get_employees(search_query="", dept_filter="", db_path="ems.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT employee_id, name, email, phone, department, designation, salary, joining_date FROM employees WHERE 1=1"
    params = []

    if search_query:
        query += " AND (employee_id LIKE ? OR name LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])

    if dept_filter and dept_filter != "All":
        query += " AND department = ?"
        params.append(dept_filter)

    query += " ORDER BY name"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "employee_id": r[0],
            "name": r[1],
            "email": r[2],
            "phone": r[3],
            "department": r[4],
            "designation": r[5],
            "salary": r[6],
            "joining_date": r[7],
        }
        for r in rows
    ]


def update_employee(employee_id, data, db_path="ems.db"):
    required = [
        "name",
        "email",
        "phone",
        "department",
        "designation",
        "salary",
        "joining_date",
    ]
    for field in required:
        if not data.get(field) or not str(data[field]).strip():
            return False, f"Field '{field}' cannot be empty."

    if not validate_email(data["email"]):
        return False, "Invalid email format."

    try:
        salary = float(data["salary"])
        if salary <= 0:
            return False, "Salary must be a positive number."
    except ValueError:
        return False, "Salary must be a valid number."

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE employees 
            SET name = ?, email = ?, phone = ?, department = ?, designation = ?, salary = ?, joining_date = ?
            WHERE employee_id = ?
        """,
            (
                str(data["name"]).strip(),
                str(data["email"]).strip(),
                str(data["phone"]).strip(),
                str(data["department"]).strip(),
                str(data["designation"]).strip(),
                salary,
                str(data["joining_date"]).strip(),
                str(employee_id).strip(),
            ),
        )
        conn.commit()
        return True, "Employee updated successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def delete_employee(employee_id, db_path="ems.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM employees WHERE employee_id = ?", (str(employee_id).strip(),)
        )
        conn.commit()
        if cursor.rowcount > 0:
            return True, "Employee deleted successfully."
        return False, "Employee not found."
    finally:
        conn.close()
