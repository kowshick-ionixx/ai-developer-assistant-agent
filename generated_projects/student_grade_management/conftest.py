import os
import sys

import pytest

# Ensure the generated project root is in sys.path so tests can import database.py
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database import init_db


@pytest.fixture(autouse=True)
def test_db():
    # Use a temporary test database file for each test
    test_db_path = os.path.join(project_root, "test_students.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Initialize test DB
    init_db(db_path=test_db_path)

    yield test_db_path

    # Cleanup after test
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
