# Simple Expense Tracker

A simple web-based Expense Tracker application built with Python, Streamlit, and SQLite.

## Features
- Add new expenses with category, amount, date, and description.
- View all recorded expenses.
- Edit or delete existing expenses.
- Dashboard with total expense calculation and category summary table & chart.
- SQLite persistence.
- Input validation (amount > 0, valid date format, non-empty category).

## Installation & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Running Tests
Run pytest in the project directory:
```bash
pytest
```
