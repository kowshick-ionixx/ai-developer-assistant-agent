import pytest
from database import add_transaction, delete_transaction, get_transactions, init_db


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_expenses.db"
    db_path = str(db_file)
    init_db(db_path)
    return db_path


def test_add_and_get_expenses(temp_db):
    add_transaction("2023-10-01", "Lunch", 50.0, "Food", "Expense", temp_db)
    df = get_transactions(temp_db)
    assert len(df) == 1
    assert df.iloc[0]["amount"] == 50.0
    assert df.iloc[0]["category"] == "Food"
    assert df.iloc[0]["date"] == "2023-10-01"
    assert df.iloc[0]["title"] == "Lunch"


def test_delete_expense(temp_db):
    add_transaction("2023-10-03", "Water", 10.0, "Utilities", "Expense", temp_db)
    df = get_transactions(temp_db)
    assert len(df) == 1
    t_id = int(df.iloc[0]["id"])

    delete_transaction(t_id, temp_db)
    df_after = get_transactions(temp_db)
    assert df_after.empty
