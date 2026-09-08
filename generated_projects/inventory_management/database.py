import sqlite3
import os

DB_NAME = "inventory.db"

def get_connection(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK(quantity >= 0),
            price REAL NOT NULL CHECK(price >= 0)
        )
    ''')
    conn.commit()
    conn.close()

def add_product(product_id, name, category, quantity, price, db_path=DB_NAME):
    if quantity < 0:
        raise ValueError("Quantity cannot be negative.")
    if price < 0:
        raise ValueError("Price cannot be negative.")
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO products (product_id, name, category, quantity, price) VALUES (?, ?, ?, ?, ?)",
            (product_id, name, category, quantity, price)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Product ID '{product_id}' already exists or invalid data.") from e
    finally:
        conn.close()

def update_product(product_id, name, category, quantity, price, db_path=DB_NAME):
    if quantity < 0:
        raise ValueError("Quantity cannot be negative.")
    if price < 0:
        raise ValueError("Price cannot be negative.")

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE products SET name = ?, category = ?, quantity = ?, price = ? WHERE product_id = ?",
        (name, category, quantity, price, product_id)
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    if rows_affected == 0:
        raise ValueError(f"Product ID '{product_id}' not found.")
    return True

def delete_product(product_id, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    if rows_affected == 0:
        raise ValueError(f"Product ID '{product_id}' not found.")
    return True

def get_all_products(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products ORDER BY product_id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def search_products(query, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE name LIKE ? OR category LIKE ? ORDER BY product_id", 
                   (f"%{query}%", f"%{query}%"))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_low_stock_products(threshold=5, db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE quantity <= ? ORDER BY quantity ASC", (threshold,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_inventory_stats(db_path=DB_NAME):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(quantity * price), SUM(quantity) FROM products")
    row = cursor.fetchone()
    conn.close()
    total_products = row[0] or 0
    total_value = row[1] or 0.0
    total_quantity = row[2] or 0
    return {
        "total_products": total_products,
        "total_value": total_value,
        "total_quantity": total_quantity
    }
