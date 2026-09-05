import os
import pytest
from database import init_db, add_task, get_tasks, complete_task, delete_task

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_todo.db"
    db_path = str(db_file)
    init_db(db_path)
    return db_path

def test_add_and_get_task(test_db):
    add_task("Buy groceries", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Buy groceries"
    assert tasks[0]["status"] == "pending"

def test_add_empty_task_raises_error(test_db):
    with pytest.raises(ValueError):
        add_task("", db_path=test_db)
    with pytest.raises(ValueError):
        add_task("   ", db_path=test_db)

def test_complete_task(test_db):
    add_task("Read a book", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    task_id = tasks[0]["id"]
    
    complete_task(task_id, db_path=test_db)
    updated_tasks = get_tasks(db_path=test_db)
    assert updated_tasks[0]["status"] == "completed"

def test_delete_task(test_db):
    add_task("Clean room", db_path=test_db)
    tasks = get_tasks(db_path=test_db)
    task_id = tasks[0]["id"]
    
    delete_task(task_id, db_path=test_db)
    assert len(get_tasks(db_path=test_db)) == 0
