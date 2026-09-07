from database import get_connection
import sqlite3

def add_department(name, db_path="ems.db"):
    if not name or not name.strip():
        return False, "Department name cannot be empty."
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO departments (name) VALUES (?)", (name.strip(),))
        conn.commit()
        return True, "Department added successfully."
    except sqlite3.IntegrityError:
        return False, "Department already exists."
    finally:
        conn.close()

def get_departments(db_path="ems.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM departments ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]

def update_department(dept_id, new_name, db_path="ems.db"):
    if not new_name or not new_name.strip():
        return False, "Department name cannot be empty."
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM departments WHERE id = ?", (dept_id,))
        row = cursor.fetchone()
        if not row:
            return False, "Department not found."
        old_name = row[0]
        
        cursor.execute("UPDATE departments SET name = ? WHERE id = ?", (new_name.strip(), dept_id))
        cursor.execute("UPDATE employees SET department = ? WHERE department = ?", (new_name.strip(), old_name))
        conn.commit()
        return True, "Department updated successfully."
    except sqlite3.IntegrityError:
        return False, "Department name already exists."
    finally:
        conn.close()

def delete_department(dept_id, db_path="ems.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM departments WHERE id = ?", (dept_id,))
        row = cursor.fetchone()
        if not row:
            return False, "Department not found."
        dept_name = row[0]
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE department = ?", (dept_name,))
        count = cursor.fetchone()[0]
        if count > 0:
            return False, f"Cannot delete department '{dept_name}' because it has {count} assigned employee(s)."
        
        cursor.execute("DELETE FROM departments WHERE id = ?", (dept_id,))
        conn.commit()
        return True, "Department deleted successfully."
    finally:
        conn.close()
