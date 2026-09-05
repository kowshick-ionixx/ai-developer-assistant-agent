# Simple HRMS Application

A lightweight Human Resource Management System built with Python, Streamlit, and SQLite.

## Features
- **Employee Registration**: Register new employees with validation (name, email, department, role, date of joining, salary).
- **Employee List & Details**: View all employees and inspect individual profiles, including attendance history and leave requests.
- **Attendance Tracking**: Record and view daily attendance (Present, Absent, Late).
- **Leave Management**: Submit leave requests and update their status (Pending, Approved, Rejected).
- **Dashboard**: Summary metrics for total employees, attendance rate, and pending leaves.

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
```bash
pytest tests/
```
