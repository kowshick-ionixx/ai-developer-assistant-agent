import sqlite3
from datetime import datetime

class ExpenseDatabase:
    def __init__(self, db_path="expenses.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                date TEXT NOT NULL,
                description TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_expense(self, category, amount, date, description=""):
        if amount <= 0:
            raise ValueError("Amount must be greater than zero.")
        if not category.strip():
            raise ValueError("Category cannot be empty.")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format.")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenses (category, amount, date, description) VALUES (?, ?, ?, ?)",
            (category.strip(), float(amount), date, description.strip())
        )
        conn.commit()
        conn.close()

    def get_expenses(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, category, amount, date, description FROM expenses ORDER BY date DESC, id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def update_expense(self, expense_id, category, amount, date, description=""):
        if amount <= 0:
            raise ValueError("Amount must be greater than zero.")
        if not category.strip():
            raise ValueError("Category cannot be empty.")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format.")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE expenses SET category = ?, amount = ?, date = ?, description = ? WHERE id = ?",
            (category.strip(), float(amount), date, description.strip(), expense_id)
        )
        conn.commit()
        conn.close()

    def delete_expense(self, expense_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
        conn.close()

    def get_total_expense(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(amount) FROM expenses")
        result = cursor.fetchone()[0]
        conn.close()
        return result if result else 0.0

    def get_category_summary(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT category, SUM(amount) FROM expenses GROUP BY category ORDER BY SUM(amount) DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
