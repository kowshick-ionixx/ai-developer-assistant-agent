import sqlite3
import os
from datetime import datetime, date

class Database:
    def __init__(self, db_path="library.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    isbn TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK(quantity >= 0),
                    available_quantity INTEGER NOT NULL CHECK(available_quantity >= 0)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS borrowings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    member_id INTEGER NOT NULL,
                    borrow_date TEXT NOT NULL,
                    due_date TEXT NOT NULL,
                    return_date TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    FOREIGN KEY(book_id) REFERENCES books(id),
                    FOREIGN KEY(member_id) REFERENCES members(id)
                )
            ''')
            conn.commit()

    # Book Operations
    def add_book(self, title, author, isbn, category, quantity):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO books (title, author, isbn, category, quantity, available_quantity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (title, author, isbn, category, quantity, quantity))
            conn.commit()
            return cursor.lastrowid

    def get_all_books(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books ORDER BY title")
            return [dict(row) for row in cursor.fetchall()]

    def get_book_by_id(self, book_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_book_by_isbn(self, isbn):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books WHERE isbn = ?", (isbn,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_book(self, book_id, title, author, isbn, category, quantity):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Calculate new available quantity adjustment
            cursor.execute("SELECT quantity, available_quantity FROM books WHERE id = ?", (book_id,))
            row = cursor.fetchone()
            if not row:
                return False
            old_qty, old_avail = row['quantity'], row['available_quantity']
            diff = quantity - old_qty
            new_avail = old_avail + diff
            if new_avail < 0:
                raise ValueError("Cannot reduce total quantity below currently borrowed count.")
            
            cursor.execute('''
                UPDATE books SET title = ?, author = ?, isbn = ?, category = ?, quantity = ?, available_quantity = ?
                WHERE id = ?
            ''', (title, author, isbn, category, quantity, new_avail, book_id))
            conn.commit()
            return True

    def delete_book(self, book_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check if book has active borrowings
            cursor.execute("SELECT COUNT(*) as count FROM borrowings WHERE book_id = ? AND status = 'active'", (book_id,))
            if cursor.fetchone()['count'] > 0:
                raise ValueError("Cannot delete a book that is currently borrowed.")
            cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
            conn.commit()
            return True

    # Member Operations
    def add_member(self, member_id, name, email, phone):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO members (member_id, name, email, phone)
                VALUES (?, ?, ?, ?)
            ''', (member_id, name, email, phone))
            conn.commit()
            return cursor.lastrowid

    def get_all_members(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM members ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]

    def get_member_by_id(self, member_id_pk):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM members WHERE id = ?", (member_id_pk,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_member_by_member_id(self, member_id_str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM members WHERE member_id = ?", (member_id_str,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_member(self, member_id_pk, member_id, name, email, phone):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE members SET member_id = ?, name = ?, email = ?, phone = ?
                WHERE id = ?
            ''', (member_id, name, email, phone, member_id_pk))
            conn.commit()
            return True

    def delete_member(self, member_id_pk):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM borrowings WHERE member_id = ? AND status = 'active'", (member_id_pk,))
            if cursor.fetchone()['count'] > 0:
                raise ValueError("Cannot delete a member with active borrowing records.")
            cursor.execute("DELETE FROM members WHERE id = ?", (member_id_pk,))
            conn.commit()
            return True

    # Borrowing Operations
    def borrow_book(self, book_id, member_id, borrow_date, due_date):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Check availability
            cursor.execute("SELECT available_quantity FROM books WHERE id = ?", (book_id,))
            book = cursor.fetchone()
            if not book or book['available_quantity'] <= 0:
                raise ValueError("Book is not available for borrowing.")
            
            # Check duplicate active borrowing for same book and member
            cursor.execute('''
                SELECT COUNT(*) as count FROM borrowings 
                WHERE book_id = ? AND member_id = ? AND status = 'active'
            ''', (book_id, member_id))
            if cursor.fetchone()['count'] > 0:
                raise ValueError("Member already has an active borrowing record for this book.")

            # Insert borrowing record
            cursor.execute('''
                INSERT INTO borrowings (book_id, member_id, borrow_date, due_date, status)
                VALUES (?, ?, ?, ?, 'active')
            ''', (book_id, member_id, borrow_date, due_date))

            # Reduce available quantity
            cursor.execute('''
                UPDATE books SET available_quantity = available_quantity - 1 WHERE id = ?
            ''', (book_id,))
            conn.commit()
            return cursor.lastrowid

    def return_book(self, borrowing_id, return_date):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM borrowings WHERE id = ? AND status = 'active'", (borrowing_id,))
            borrowing = cursor.fetchone()
            if not borrowing:
                raise ValueError("Active borrowing record not found.")

            book_id = borrowing['book_id']

            # Update borrowing record
            cursor.execute('''
                UPDATE borrowings SET return_date = ?, status = 'returned' WHERE id = ?
            ''', (return_date, borrowing_id))

            # Increase available quantity
            cursor.execute('''
                UPDATE books SET available_quantity = available_quantity + 1 WHERE id = ?
            ''', (book_id,))
            conn.commit()
            return True

    def get_all_borrowings(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT b.*, bk.title as book_title, bk.isbn, m.name as member_name, m.member_id as member_code
                FROM borrowings b
                JOIN books bk ON b.book_id = bk.id
                JOIN members m ON b.member_id = m.id
                ORDER BY b.borrow_date DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]
