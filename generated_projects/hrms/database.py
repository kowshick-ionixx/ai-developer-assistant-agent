import os
import sqlite3
from datetime import datetime

DB_NAME = os.path.join(os.path.dirname(__file__), "hrms.db")


def get_connection(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            role TEXT NOT NULL,
            date_joined TEXT NOT NULL
        )
    """)

    # Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
    """)

    # Leaves table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
    """)

    conn.commit()
    conn.close()


def add_employee(name, email, department, role, date_joined=None, db_path=DB_NAME):
    if not date_joined:
        date_joined = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO employees (name, email, department, role, date_joined) VALUES (?, ?, ?, ?, ?)",
            (name, email, department, role, date_joined),
        )
        conn.commit()
        return True, "Employee added successfully."
    except sqlite3.IntegrityError:
        return False, "An employee with this email already exists."
    finally:
        conn.close()


def get_all_employees(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_attendance(employee_id, date, status, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    # Check if already marked for date
    cursor.execute(
        "SELECT id FROM attendance WHERE employee_id = ? AND date = ?",
        (employee_id, date),
    )
    existing = cursor.fetchone()
    if existing:
        cursor.execute(
            "UPDATE attendance SET status = ? WHERE id = ?", (status, existing["id"])
        )
    else:
        cursor.execute(
            "INSERT INTO attendance (employee_id, date, status) VALUES (?, ?, ?)",
            (employee_id, date, status),
        )
    conn.commit()
    conn.close()
    return True, "Attendance recorded."


def get_attendance(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT employees.name, attendance.date, attendance.status 
        FROM attendance 
        JOIN employees ON attendance.employee_id = employees.id
        ORDER BY attendance.date DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def apply_leave(employee_id, start_date, end_date, reason, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO leaves (employee_id, start_date, end_date, reason, status) VALUES (?, ?, ?, ?, 'Pending')",
        (employee_id, start_date, end_date, reason),
    )
    conn.commit()
    conn.close()
    return True, "Leave application submitted."


def get_leaves(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT leaves.id, employees.name, leaves.start_date, leaves.end_date, leaves.reason, leaves.status 
        FROM leaves 
        JOIN employees ON leaves.employee_id = employees.id
        ORDER BY leaves.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_leave_status(leave_id, status, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE leaves SET status = ? WHERE id = ?", (status, leave_id))
    conn.commit()
    conn.close()
    return True, f"Leave status updated to {status}."
