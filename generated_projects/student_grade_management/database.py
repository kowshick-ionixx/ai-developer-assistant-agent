import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "students.db")


def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            class_name TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            subject TEXT NOT NULL,
            mark REAL NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    """)
    conn.commit()
    conn.close()


def add_student(name, roll_number, class_name, db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO students (name, roll_number, class_name) VALUES (?, ?, ?)",
            (name, roll_number, class_name),
        )
        conn.commit()
        return True, "Student added successfully."
    except sqlite3.IntegrityError:
        return False, "Roll number already exists."
    finally:
        conn.close()


def get_all_students(db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number, class_name FROM students")
    rows = cursor.fetchall()
    conn.close()
    return rows


def add_mark(student_id, subject, mark, db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO marks (student_id, subject, mark) VALUES (?, ?, ?)",
        (student_id, subject, mark),
    )
    conn.commit()
    conn.close()


def get_marks_for_student(student_id, db_path=DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subject, mark FROM marks WHERE student_id = ?", (student_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def calculate_grade(average):
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"
