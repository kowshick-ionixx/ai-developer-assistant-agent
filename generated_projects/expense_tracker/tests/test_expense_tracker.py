import os
import pytest
from database import ExpenseDatabase

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_expenses.db"
    db = ExpenseDatabase(db_path=str(db_file))
    return db

def test_add_and_get_expense(temp_db):
    temp_db.add_expense("Food", 25.50, "2023-10-01", "Lunch")
    expenses = temp_db.get_expenses()
    assert len(expenses) == 1
    assert expenses[0][1] == "Food"
    assert expenses[0][2] == 25.50
    assert expenses[0][3] == "2023-10-01"
    assert expenses[0][4] == "Lunch"

def test_validation_rules(temp_db):
    with pytest.raises(ValueError):
        temp_db.add_expense("Food", -10.0, "2023-10-01")
    with pytest.raises(ValueError):
        temp_db.add_expense("", 10.0, "2023-10-01")
    with pytest.raises(ValueError):
        temp_db.add_expense("Food", 10.0, "invalid-date")

def test_total_and_summary(temp_db):
    temp_db.add_expense("Food", 20.0, "2023-10-01")
    temp_db.add_expense("Food", 30.0, "2023-10-02")
    temp_db.add_expense("Transport", 15.0, "2023-10-01")
    
    assert temp_db.get_total_expense() == 65.0
    
    summary = temp_db.get_category_summary()
    assert len(summary) == 2
    assert summary[0][0] == "Food"
    assert summary[0][1] == 50.0
    assert summary[1][0] == "Transport"
    assert summary[1][1] == 15.0

def test_update_and_delete(temp_db):
    temp_db.add_expense("Food", 20.0, "2023-10-01")
    expenses = temp_db.get_expenses()
    exp_id = expenses[0][0]
    
    temp_db.update_expense(exp_id, "Utilities", 50.0, "2023-10-02", "Electricity")
    updated = temp_db.get_expenses()
    assert updated[0][1] == "Utilities"
    assert updated[0][2] == 50.0
    assert updated[0][3] == "2023-10-02"
    assert updated[0][4] == "Electricity"
    
    temp_db.delete_expense(exp_id)
    assert len(temp_db.get_expenses()) == 0
