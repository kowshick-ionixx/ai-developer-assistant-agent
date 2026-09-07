import pandas as pd
import streamlit as st
from database import init_db
from department import (
    add_department,
    delete_department,
    get_departments,
    update_department,
)
from employee import add_employee, delete_employee, get_employees, update_employee

st.set_page_config(
    page_title="Employee Management System", page_icon="👥", layout="wide"
)

# Initialize database
init_db()

st.title("👥 Employee Management System")

# Sidebar navigation
menu = ["Dashboard", "Employees", "Departments"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Dashboard":
    st.header("📊 Dashboard & Summary")

    employees = get_employees()
    departments = get_departments()

    total_employees = len(employees)
    total_departments = len(departments)

    avg_salary = (
        sum(e["salary"] for e in employees) / total_employees
        if total_employees > 0
        else 0.0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", total_employees)
    col2.metric("Total Departments", total_departments)
    col3.metric("Average Salary ($)", f"{avg_salary:,.2f}")

    st.subheader("Department-wise Employee Count")
    if employees:
        df_emp = pd.DataFrame(employees)
        dept_counts = df_emp["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Employee Count"]
        st.bar_chart(dept_counts.set_index("Department"))
        st.dataframe(dept_counts, use_container_width=True)
    else:
        st.info("No employee data available for chart.")

elif choice == "Employees":
    st.header("👤 Employee Management")

    tab1, tab2 = st.tabs(["Employee List & Search", "Add New Employee"])

    departments = get_departments()
    dept_names = [d["name"] for d in departments]

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("🔍 Search by Name or ID")
        with col2:
            dept_filter_options = ["All"] + dept_names
            dept_filter = st.selectbox("Filter by Department", dept_filter_options)

        employees = get_employees(search_query=search_query, dept_filter=dept_filter)

        if employees:
            df = pd.DataFrame(employees)
            st.dataframe(df, use_container_width=True)

            st.subheader("Edit or Delete Employee")
            emp_ids = [e["employee_id"] for e in employees]
            selected_emp_id = st.selectbox("Select Employee ID", emp_ids)

            emp_data = next(
                (e for e in employees if e["employee_id"] == selected_emp_id), None
            )

            if emp_data:
                with st.form("edit_employee_form"):
                    e_name = st.text_input("Name", value=emp_data["name"])
                    e_email = st.text_input("Email", value=emp_data["email"])
                    e_phone = st.text_input("Phone", value=emp_data["phone"])

                    current_dept_idx = (
                        dept_names.index(emp_data["department"])
                        if emp_data["department"] in dept_names
                        else 0
                    )
                    e_dept = (
                        st.selectbox("Department", dept_names, index=current_dept_idx)
                        if dept_names
                        else st.text_input("Department", value=emp_data["department"])
                    )

                    e_desig = st.text_input(
                        "Designation", value=emp_data["designation"]
                    )
                    e_salary = st.number_input(
                        "Salary ($)",
                        value=float(emp_data["salary"]),
                        min_value=0.0,
                        step=100.0,
                    )
                    e_joining = st.text_input(
                        "Joining Date (YYYY-MM-DD)", value=emp_data["joining_date"]
                    )

                    col_update, col_delete = st.columns(2)
                    submitted_update = col_update.form_submit_button("Update Employee")
                    submitted_delete = col_delete.form_submit_button("Delete Employee")

                    if submitted_update:
                        updated_info = {
                            "name": e_name,
                            "email": e_email,
                            "phone": e_phone,
                            "department": e_dept if isinstance(e_dept, str) else e_dept,
                            "designation": e_desig,
                            "salary": e_salary,
                            "joining_date": e_joining,
                        }
                        success, msg = update_employee(selected_emp_id, updated_info)
                        if success:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)

                    if submitted_delete:
                        success, msg = delete_employee(selected_emp_id)
                        if success:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No employees found.")

    with tab2:
        if not dept_names:
            st.warning("Please create at least one department before adding employees.")
        else:
            with st.form("add_employee_form", clear_on_submit=True):
                new_id = st.text_input("Employee ID")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email Address")
                new_phone = st.text_input("Phone Number")
                new_dept = st.selectbox("Department", dept_names)
                new_desig = st.text_input("Designation")
                new_salary = st.number_input("Salary ($)", min_value=0.0, step=100.0)
                new_joining = st.text_input(
                    "Joining Date (YYYY-MM-DD)", placeholder="2023-01-15"
                )

                submitted = st.form_submit_button("Add Employee")
                if submitted:
                    emp_dict = {
                        "employee_id": new_id,
                        "name": new_name,
                        "email": new_email,
                        "phone": new_phone,
                        "department": new_dept,
                        "designation": new_desig,
                        "salary": new_salary,
                        "joining_date": new_joining,
                    }
                    success, msg = add_employee(emp_dict)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

elif choice == "Departments":
    st.header("🏢 Department Management")

    tab1, tab2 = st.tabs(["View & Manage Departments", "Add Department"])

    with tab1:
        departments = get_departments()
        if departments:
            for dept in departments:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(f"ID: {dept['id']} | Name: {dept['name']}")
                with col2:
                    new_dept_name = st.text_input(
                        f"Rename {dept['name']}",
                        value=dept["name"],
                        key=f"rename_{dept['id']}",
                    )
                with col3:
                    if st.button("Update", key=f"btn_update_{dept['id']}"):
                        success, msg = update_department(dept["id"], new_dept_name)
                        if success:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                    if st.button("Delete", key=f"btn_delete_{dept['id']}"):
                        success, msg = delete_department(dept["id"])
                        if success:
                            st.success(msg)
                            st.experimental_rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No departments found.")

    with tab2, st.form("add_department_form", clear_on_submit=True):
        dept_name = st.text_input("Department Name")
        submitted = st.form_submit_button("Create Department")
        if submitted:
            success, msg = add_department(dept_name)
            if success:
                st.success(msg)
            else:
                st.error(msg)
