import sqlite3

DB_NAME = "ems.db"


def get_connection(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            department TEXT NOT NULL,
            designation TEXT NOT NULL,
            salary REAL NOT NULL,
            joining_date TEXT NOT NULL,
            FOREIGN KEY (department) REFERENCES departments(name) ON UPDATE CASCADE
        )
    """)

    conn.commit()
    conn.close()
