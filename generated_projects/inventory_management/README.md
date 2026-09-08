# Inventory Management System

A simple, lightweight Inventory Management System built with Python, Streamlit, and SQLite.

## Features
- Add, edit, and delete products
- Prevent duplicate Product IDs and negative quantities/prices
- Search products by name or category
- Dashboard with total items, inventory value, and low-stock alerts
- SQLite database storage

## Installation & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```
