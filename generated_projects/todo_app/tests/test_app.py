import os
import sqlite3
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


def test_init_db(tmp_path):
    db_path = tmp_path / "test_todos.db"
    app.DB_FILE = str(db_path)
    app.init_db()
    assert os.path.exists(db_path)


def test_add_and_get_todos(tmp_path):
    db_path = tmp_path / "test_todos.db"
    app.DB_FILE = str(db_path)
    app.init_db()

    app.add_todo("Test task 1")
    todos = app.get_todos()
    assert len(todos) == 1
    assert todos[0][1] == "Test task 1"
    assert todos[0][2] == 0


def test_toggle_and_delete_todo(tmp_path):
    db_path = tmp_path / "test_todos.db"
    app.DB_FILE = str(db_path)
    app.init_db()

    app.add_todo("Task to delete")
    todos = app.get_todos()
    todo_id = todos[0][0]

    app.toggle_todo(todo_id, True)
    todos = app.get_todos()
    assert todos[0][2] == 1

    app.delete_todo(todo_id)
    todos = app.get_todos()
    assert len(todos) == 0
