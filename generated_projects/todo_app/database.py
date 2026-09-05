import sqlite3
import os

def get_connection(db_path="todo.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path="todo.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()

def add_task(title, db_path="todo.db"):
    if not title or not title.strip():
        raise ValueError("Task title cannot be empty.")
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tasks (title, status) VALUES (?, 'pending')", (title.strip(),))
    conn.commit()
    conn.close()

def get_tasks(db_path="todo.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, status FROM tasks")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def complete_task(task_id, db_path="todo.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()

def delete_task(task_id, db_path="todo.db"):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
