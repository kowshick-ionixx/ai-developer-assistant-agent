import os
import sqlite3

import pytest
from database import Database
from library import LibraryManager

TEST_DB = "test_library.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass


def test_init_db():
    db = Database(TEST_DB)
    db.init_db()
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='books';"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_book_crud():
    db = Database(TEST_DB)
    db.init_db()
    db.add_book("Python 101", "Alice", "1234567890", "Tech", 3)
    books = db.get_all_books()
    assert len(books) == 1
    book_id = books[0]["id"]

    db.update_book(book_id, "Python 101 Advanced", "Alice", "1234567890", "Tech", 5)
    book = db.get_book_by_id(book_id)
    assert book["title"] == "Python 101 Advanced"
    assert book["available_qty"] == 5

    db.delete_book(book_id)
    assert len(db.get_all_books()) == 0


def test_member_crud():
    db = Database(TEST_DB)
    db.init_db()
    db.add_member("M001", "Bob Smith", "bob@example.com", "555-1234")
    members = db.get_all_members()
    assert len(members) == 1
    member_pk = members[0]["id"]

    db.update_member(member_pk, "M001", "Bob Builder", "bob@builder.com", "555-4321")
    member = db.get_member_by_id(member_pk)
    assert member["name"] == "Bob Builder"
    assert member["email"] == "bob@builder.com"

    db.delete_member(member_pk)
    assert len(db.get_all_members()) == 0


def test_borrow_and_return():
    manager = LibraryManager(TEST_DB)
    manager.db.init_db()
    manager.add_book("Clean Code", "Robert Martin", "111", "Programming", 2)
    member_pk = manager.register_member(
        "M002", "Charlie", "charlie@example.com", "555-9999"
    )

    books = manager.get_all_books()
    book_id = books[0]["id"]

    success, _msg = manager.borrow_book(member_pk, book_id)
    assert success is True

    book = manager.db.get_book_by_id(book_id)
    assert book["available_qty"] == 1

    borrowings = manager.get_all_borrowings()
    active_borrowing = [b for b in borrowings if b["return_date"] is None]
    assert len(active_borrowing) == 1
    borrowing_id = active_borrowing[0]["id"]

    success, _msg = manager.return_book(borrowing_id)
    assert success is True

    book = manager.db.get_book_by_id(book_id)
    assert book["available_qty"] == 2


def test_unavailable_book_prevention():
    manager = LibraryManager(TEST_DB)
    manager.db.init_db()
    manager.add_book("Rare Book", "Unknown", "222", "History", 0)
    member_pk = manager.register_member(
        "M003", "David", "david@example.com", "555-0000"
    )

    books = manager.get_all_books()
    book_id = books[0]["id"]

    success, msg = manager.borrow_book(member_pk, book_id)
    assert success is False
    assert "not available" in msg.lower()


def test_duplicate_borrow_prevention():
    manager = LibraryManager(TEST_DB)
    manager.db.init_db()
    manager.add_book("Python Guide", "Guido", "999", "Tech", 5)
    member_pk = manager.register_member(
        "M004", "Alice", "alice@example.com", "555-1111"
    )

    books = manager.get_all_books()
    book_id = books[0]["id"]

    success, msg = manager.borrow_book(member_pk, book_id)
    assert success is True

    success, msg = manager.borrow_book(member_pk, book_id)
    assert success is False
    assert "already" in msg.lower()


def test_search_and_filtering():
    manager = LibraryManager(TEST_DB)
    manager.db.init_db()
    manager.add_book("Python Unique Programming", "Guido", "123", "Tech", 3)
    manager.add_book("Django Web Development", "Jacob", "456", "Web", 2)

    results = manager.search_books("Unique")
    assert len(results) == 1
    assert results[0]["title"] == "Python Unique Programming"

    filtered = manager.filter_books_by_category("Web")
    assert len(filtered) == 1
    assert filtered[0]["isbn"] == "456"


def test_dashboard_stats():
    manager = LibraryManager(TEST_DB)
    manager.db.init_db()
    # Ensure fresh DB has no books from other tests
    manager.add_book("Book Unique A", "Author A", "333", "Fiction", 5)
    manager.register_member("M005", "Eve", "eve@example.com", "555-6666")
    stats = manager.get_dashboard_stats()
    assert stats["total_book_titles"] == 1
    assert stats["total_books"] == 5
    assert stats["total_members"] == 1
