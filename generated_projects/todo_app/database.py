import sqlite3
from datetime import datetime

DB_NAME = "todo.db"


def init_db(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            created_date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_task(description, db_path=DB_NAME):
    if not description or not description.strip():
        raise ValueError("Task description cannot be empty.")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO tasks (description, completed, created_date) VALUES (?, 0, ?)",
        (description.strip(), created_date),
    )
    conn.commit()
    conn.close()


def get_tasks(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, description, completed, created_date FROM tasks ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "description": r[1], "completed": bool(r[2]), "created_date": r[3]}
        for r in rows
    ]


def complete_task(task_id, db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def delete_task(task_id, db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def get_task_counts(db_path=DB_NAME):
    tasks = get_tasks(db_path)
    total = len(tasks)
    completed = sum(1 for t in tasks if t["completed"])
    pending = total - completed
    return {"total": total, "completed": completed, "pending": pending}
