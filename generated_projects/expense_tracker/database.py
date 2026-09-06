import sqlite3
import os
from datetime import datetime

class ExpenseDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "expenses.db")
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self.get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    date TEXT NOT NULL,
                    description TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def add_expense(self, amount, category, date, description):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO expenses (amount, category, date, description) VALUES (?, ?, ?, ?)",
                (float(amount), category, date, description)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_expenses(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM expenses ORDER BY date DESC, id DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def update_expense(self, expense_id, amount, category, date, description):
        conn = self.get_connection()
        try:
            conn.execute(
                "UPDATE expenses SET amount = ?, category = ?, date = ?, description = ? WHERE id = ?",
                (float(amount), category, date, description, int(expense_id))
            )
            conn.commit()
        finally:
            conn.close()

    def delete_expense(self, expense_id):
        conn = self.get_connection()
        try:
            conn.execute("DELETE FROM expenses WHERE id = ?", (int(expense_id),))
            conn.commit()
        finally:
            conn.close()

    def get_category_totals(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT category, SUM(amount) as total 
                FROM expenses 
                GROUP BY category 
                ORDER BY total DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
