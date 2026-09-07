def test_add_task():
    tasks = []
    title = "Test Task"
    description = "Test Desc"
    if title.strip():
        tasks.append({"title": title, "description": description, "completed": False})
    
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Test Task"
    assert tasks[0]["completed"] is False

def test_toggle_task():
    tasks = [{"title": "Task 1", "description": "", "completed": False}]
    # Toggle complete
    tasks[0]["completed"] = not tasks[0]["completed"]
    assert tasks[0]["completed"] is True
    # Toggle back
    tasks[0]["completed"] = not tasks[0]["completed"]
    assert tasks[0]["completed"] is False

def test_delete_task():
    tasks = [{"title": "Task 1", "description": "", "completed": False}]
    tasks.pop(0)
    assert len(tasks) == 0
