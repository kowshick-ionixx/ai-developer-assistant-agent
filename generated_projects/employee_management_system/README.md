# Employee Management System

A web-based Employee Management System built with Python, Streamlit, and SQLite.

## Features
- **Employee Management**: Add, view, edit, delete, search by name/ID, and filter by department.
- **Department Management**: Create, view, edit, and delete departments (deletion restricted if employees are assigned).
- **Dashboard**: Summary metrics (total employees, departments, average salary) and department-wise employee count visualization.
- **Validation**: Unique employee IDs, email format validation, positive salary checks, and required field validation.

## Installation & Running
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Running Tests
```bash
pytest tests/
```
