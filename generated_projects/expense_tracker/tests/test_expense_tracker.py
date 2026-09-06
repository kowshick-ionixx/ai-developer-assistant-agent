import os
import tempfile
import pytest
from database import ExpenseDatabase

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    db = ExpenseDatabase(db_path=path)
    yield db
    os.unlink(path)

def test_add_and_get_expenses(temp_db):
    temp_db.add_expense(50.0, "Food", "2023-10-01", "Lunch")
    expenses = temp_db.get_expenses()
    assert len(expenses) == 1
    assert expenses[0]['amount'] == 50.0
    assert expenses[0]['category'] == "Food"
    assert expenses[0]['date'] == "2023-10-01"
    assert expenses[0]['description'] == "Lunch"

def test_update_expense(temp_db):
    exp_id = temp_db.add_expense(20.0, "Transport", "2023-10-02", "Bus")
    temp_db.update_expense(exp_id, 25.0, "Transport", "2023-10-02", "Subway")
    
    expenses = temp_db.get_expenses()
    assert len(expenses) == 1
    assert expenses[0]['amount'] == 25.0
    assert expenses[0]['description'] == "Subway"

def test_delete_expense(temp_db):
    exp_id = temp_db.add_expense(10.0, "Utilities", "2023-10-03", "Water")
    assert len(temp_db.get_expenses()) == 1
    
    temp_db.delete_expense(exp_id)
    assert len(temp_db.get_expenses()) == 0

def test_category_totals(temp_db):
    temp_db.add_expense(30.0, "Food", "2023-10-01", "Breakfast")
    temp_db.add_expense(20.0, "Food", "2023-10-02", "Dinner")
    temp_db.add_expense(100.0, "Housing", "2023-10-01", "Rent")
    
    totals = temp_db.get_category_totals()
    totals_dict = {item['category']: item['total'] for item in totals}
    
    assert totals_dict["Food"] == 50.0
    assert totals_dict["Housing"] == 100.0
