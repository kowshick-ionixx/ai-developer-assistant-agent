import sqlite3
from datetime import date
from database import Database

class LibraryManager:
    def __init__(self, db_path="library.db"):
        self.db = Database(db_path)

    def add_book(self, title, author, isbn, category, total_qty):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO books (title, author, isbn, category, total_qty, available_qty)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (title, author, isbn, category, total_qty, total_qty))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_all_books(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, author, isbn, category, total_qty, available_qty FROM books")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_book(self, book_id, title, author, isbn, category, total_qty):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT total_qty, available_qty FROM books WHERE id = ?", (book_id,))
            row = cursor.fetchone()
            if not row:
                return False
            old_total, old_available = row["total_qty"], row["available_qty"]
            diff = total_qty - old_total
            new_available = max(0, old_available + diff)
            cursor.execute("""
                UPDATE books SET title = ?, author = ?, isbn = ?, category = ?, total_qty = ?, available_qty = ?
                WHERE id = ?
            """, (title, author, isbn, category, total_qty, new_available, book_id))
            conn.commit()
            return True

    def delete_book(self, book_id):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM borrowings WHERE book_id = ? AND return_date IS NULL", (book_id,))
            if cursor.fetchone()["count"] > 0:
                return False, "Cannot delete book currently on active loan."
            cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
            conn.commit()
            return True, "Book deleted successfully."

    def search_books(self, query):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            q = f"%{query}%"
            cursor.execute("""
                SELECT id, title, author, isbn, category, total_qty, available_qty FROM books
                WHERE title LIKE ? OR author LIKE ? OR isbn LIKE ?
            """, (q, q, q))
            return [dict(row) for row in cursor.fetchall()]

    def filter_books_by_category(self, category):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, author, isbn, category, total_qty, available_qty FROM books
                WHERE category = ?
            """, (category,))
            return [dict(row) for row in cursor.fetchall()]

    def register_member(self, member_id, name, email, phone):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO members (member_id, name, email, phone)
                    VALUES (?, ?, ?, ?)
                """, (member_id, name, email, phone))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    def get_all_members(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, member_id, name, email, phone FROM members")
            return [dict(row) for row in cursor.fetchall()]

    def update_member(self, member_pk, member_id, name, email, phone):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE members SET member_id = ?, name = ?, email = ?, phone = ?
                    WHERE id = ?
                """, (member_id, name, email, phone, member_pk))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def delete_member(self, member_pk):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM borrowings WHERE member_id = ? AND return_date IS NULL", (member_pk,))
            if cursor.fetchone()["count"] > 0:
                return False, "Cannot delete member with active borrowing records."
            cursor.execute("DELETE FROM members WHERE id = ?", (member_pk,))
            conn.commit()
            return True, "Member deleted successfully."

    def search_members(self, query):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            q = f"%{query}%"
            cursor.execute("""
                SELECT id, member_id, name, email, phone FROM members
                WHERE member_id LIKE ? OR name LIKE ?
            """, (q, q))
            return [dict(row) for row in cursor.fetchall()]

    def borrow_book(self, member_id, book_id, borrow_date=None, due_date=None):
        if borrow_date is None:
            borrow_date = date.today().isoformat()
        if due_date is None:
            from datetime import timedelta
            due_date = (date.today() + timedelta(days=14)).isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Check book availability
            cursor.execute("SELECT available_qty FROM books WHERE id = ?", (book_id,))
            book_row = cursor.fetchone()
            if not book_row or book_row["available_qty"] <= 0:
                return False, "Book is not available for borrowing."

            # Check duplicate active borrowing
            cursor.execute("""
                SELECT id FROM borrowings WHERE member_id = ? AND book_id = ? AND return_date IS NULL
            """, (member_id, book_id))
            if cursor.fetchone():
                return False, "Member already has an active borrowing record for this book."

            # Insert borrowing
            cursor.execute("""
                INSERT INTO borrowings (member_id, book_id, borrow_date, due_date)
                VALUES (?, ?, ?, ?)
            """, (member_id, book_id, borrow_date, due_date))

            # Reduce available quantity
            cursor.execute("""
                UPDATE books SET available_qty = available_qty - 1 WHERE id = ?
            """, (book_id,))
            conn.commit()
            return True, "Book borrowed successfully."

    def return_book(self, borrowing_id, return_date=None):
        if return_date is None:
            return_date = date.today().isoformat()

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT book_id, return_date FROM borrowings WHERE id = ?
            """, (borrowing_id,))
            row = cursor.fetchone()
            if not row:
                return False, "Borrowing record not found."
            if row["return_date"] is not None:
                return False, "Book has already been returned."

            book_id = row["book_id"]

            cursor.execute("""
                UPDATE borrowings SET return_date = ? WHERE id = ?
            """, (return_date, borrowing_id))

            cursor.execute("""
                UPDATE books SET available_qty = available_qty + 1 WHERE id = ?
            """, (book_id,))
            conn.commit()
            return True, "Book returned successfully."

    def get_all_borrowings(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT b.id, b.member_id, b.book_id, b.borrow_date, b.due_date, b.return_date,
                       m.name as member_name, bk.title as book_title
                FROM borrowings b
                JOIN members m ON b.member_id = m.id
                JOIN books bk ON b.book_id = bk.id
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_overdue_borrowings(self, current_date=None):
        if current_date is None:
            current_date = date.today().isoformat()
        borrowings = self.get_all_borrowings()
        overdue = []
        for b in borrowings:
            if b["return_date"] is None and b["due_date"] < current_date:
                overdue.append(b)
        return overdue

    def get_dashboard_stats(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM books")
            total_titles = cursor.fetchone()["cnt"]

            cursor.execute("SELECT SUM(total_qty) as total, SUM(available_qty) as avail FROM books")
            row = cursor.fetchone()
            total_books = row["total"] or 0
            available_books = row["avail"] or 0

            cursor.execute("SELECT COUNT(*) as cnt FROM members")
            total_members = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM borrowings WHERE return_date IS NULL")
            active_loans = cursor.fetchone()["cnt"]

            today = date.today().isoformat()
            cursor.execute("SELECT COUNT(*) as cnt FROM borrowings WHERE return_date IS NULL AND due_date < ?", (today,))
            overdue_loans = cursor.fetchone()["cnt"]

            return {
                "total_book_titles": total_titles,
                "total_books": total_books,
                "available_books": available_books,
                "total_members": total_members,
                "active_loans": active_loans,
                "overdue_loans": overdue_loans
            }
