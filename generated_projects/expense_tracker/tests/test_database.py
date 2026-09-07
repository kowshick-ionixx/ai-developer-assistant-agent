import pytest
from database import add_transaction, delete_transaction, get_transactions, init_db


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_expenses.db"
    db_path = str(db_file)
    init_db(db_path)
    return db_path


def test_init_db(test_db):
    df = get_transactions(test_db)
    assert df.empty


def test_add_and_get_transaction(test_db):
    add_transaction("2023-10-01", "Groceries", 50.0, "Food", "Expense", test_db)
    df = get_transactions(test_db)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Groceries"
    assert df.iloc[0]["amount"] == 50.0
    assert df.iloc[0]["category"] == "Food"
    assert df.iloc[0]["type"] == "Expense"


def test_delete_transaction(test_db):
    add_transaction("2023-10-01", "Salary", 2000.0, "Salary", "Income", test_db)
    df = get_transactions(test_db)
    assert len(df) == 1
    t_id = int(df.iloc[0]["id"])

    delete_transaction(t_id, test_db)
    df_after = get_transactions(test_db)
    assert df_after.empty
