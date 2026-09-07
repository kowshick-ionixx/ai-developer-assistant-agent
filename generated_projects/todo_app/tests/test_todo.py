import os
import pytest
from database import init_db, add_task, get_tasks, complete_task, delete_task, get_task_counts

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_todo.db"
    init_db(str(db_file))
    return str(db_file)

def test_add_task(test_db):
    add_task("Test task 1", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    assert len(tasks) == 1
    assert tasks[0]["description"] == "Test task 1"
    assert tasks[0]["completed"] is False

def test_empty_task_validation(test_db):
    with pytest.raises(ValueError, match="Task description cannot be empty"):
        add_task("", db_path=test_db)
    with pytest.raises(ValueError, match="Task description cannot be empty"):
        add_task("   ", db_path=test_db)

def test_view_tasks(test_db):
    add_task("Task A", db_path=test_db)
    add_task("Task B", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    assert len(tasks) == 2
    descriptions = [t["description"] for t in tasks]
    assert "Task A" in descriptions
    assert "Task B" in descriptions

def test_complete_task(test_db):
    add_task("Incomplete task", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    task_id = tasks[0]["id"]
    
    complete_task(task_id, db_path=test_db)
    updated_tasks = get_tasks(db_path=test_db)
    assert updated_tasks[0]["completed"] is True

def test_delete_task(test_db):
    add_task("Task to delete", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    task_id = tasks[0]["id"]
    
    delete_task(task_id, db_path=test_db)
    remaining_tasks = get_tasks(db_path=test_db)
    assert len(remaining_tasks) == 0

def test_task_counts(test_db):
    add_task("Task 1", db_path=test_db)
    add_task("Task 2", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    complete_task(tasks[0]["id"], db_path=test_db)
    
    counts = get_task_counts(db_path=test_db)
    assert counts["total"] == 2
    assert counts["completed"] == 1
    assert counts["pending"] == 1
