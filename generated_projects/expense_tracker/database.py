import sqlite3
import pandas as pd
import os

DB_NAME = "expenses.db"

def get_connection(db_path=DB_NAME):
    return sqlite3.connect(db_path)

def init_db(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            title TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_transaction(date, title, amount, category, t_type, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions (date, title, amount, category, type)
        VALUES (?, ?, ?, ?, ?)
    ''', (date, title, amount, category, t_type))
    conn.commit()
    conn.close()

def get_transactions(db_path=DB_NAME):
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC", conn)
    conn.close()
    return df

def delete_transaction(t_id, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions WHERE id = ?", (t_id,))
    conn.commit()
    conn.close()
