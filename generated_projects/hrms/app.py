from datetime import date

import streamlit as st
from database import (
    add_employee,
    apply_leave,
    get_all_employees,
    get_attendance,
    get_leaves,
    init_db,
    mark_attendance,
    update_leave_status,
)

# Initialize database on startup
init_db()

st.set_page_config(page_title="Simple HRMS", page_icon="👥", layout="wide")

st.title("👥 Simple Human Resource Management System")

# Sidebar navigation
menu = ["Dashboard", "Employees", "Attendance", "Leave Management"]
choice = st.sidebar.selectbox("Navigation", menu)

employees = get_all_employees()

if choice == "Dashboard":
    st.header("Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(employees))
    with col2:
        leaves = get_leaves()
        pending_leaves = sum(1 for l in leaves if l["status"] == "Pending")
        st.metric("Pending Leave Requests", pending_leaves)
    with col3:
        attendance_logs = get_attendance()
        st.metric("Total Attendance Records", len(attendance_logs))

    st.subheader("Recent Employees")
    if employees:
        st.dataframe(employees[-5:], use_container_width=True)
    else:
        st.info("No employees registered yet.")

elif choice == "Employees":
    st.header("Employee Management")

    tab1, tab2 = st.tabs(["Employee Directory", "Add Employee"])

    with tab1:
        st.subheader("All Employees")
        if employees:
            st.dataframe(employees, use_container_width=True)
        else:
            st.info("No employees found.")

    with tab2:
        st.subheader("Register New Employee")
        with st.form("employee_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            department = st.selectbox(
                "Department", ["Engineering", "HR", "Sales", "Marketing", "Finance"]
            )
            role = st.text_input("Role / Job Title")
            date_joined = st.date_input("Date Joined", value=date.today())

            submitted = st.form_submit_button("Add Employee")
            if submitted:
                if name and email:
                    success, msg = add_employee(
                        name, email, department, role, str(date_joined)
                    )
                    if success:
                        st.success(msg)
                        employees = get_all_employees()  # refresh
                    else:
                        st.error(msg)
                else:
                    st.warning("Please provide Name and Email.")

elif choice == "Attendance":
    st.header("Attendance Tracking")

    if not employees:
        st.warning("Please add employees before recording attendance.")
    else:
        tab1, tab2 = st.tabs(["Mark Attendance", "Attendance History"])

        with tab1:
            with st.form("attendance_form"):
                emp_dict = {f"{e['name']} ({e['email']})": e["id"] for e in employees}
                selected_emp_label = st.selectbox(
                    "Select Employee", list(emp_dict.keys())
                )
                att_date = st.date_input("Date", value=date.today())
                status = st.selectbox(
                    "Status", ["Present", "Absent", "Half-Day", "On Leave"]
                )

                submitted = st.form_submit_button("Submit Attendance")
                if submitted:
                    emp_id = emp_dict[selected_emp_label]
                    success, msg = mark_attendance(emp_id, str(att_date), status)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        with tab2:
            st.subheader("Attendance Logs")
            logs = get_attendance()
            if logs:
                st.dataframe(logs, use_container_width=True)
            else:
                st.info("No attendance records found.")

elif choice == "Leave Management":
    st.header("Leave Management")

    if not employees:
        st.warning("Please add employees before applying for leave.")
    else:
        tab1, tab2 = st.tabs(["Apply for Leave", "Manage Leave Requests"])

        with tab1:
            with st.form("leave_form"):
                emp_dict = {f"{e['name']} ({e['email']})": e["id"] for e in employees}
                selected_emp_label = st.selectbox(
                    "Select Employee", list(emp_dict.keys())
                )
                start_date = st.date_input("Start Date", value=date.today())
                end_date = st.date_input("End Date", value=date.today())
                reason = st.text_area("Reason for Leave")

                submitted = st.form_submit_button("Submit Leave Request")
                if submitted:
                    if reason:
                        emp_id = emp_dict[selected_emp_label]
                        success, msg = apply_leave(
                            emp_id, str(start_date), str(end_date), reason
                        )
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please provide a reason for leave.")

        with tab2:
            st.subheader("Leave Requests")
            leaves = get_leaves()
            if leaves:
                for leave in leaves:
                    with st.container():
                        cols = st.columns([3, 2, 2, 2])
                        cols[0].write(f"**{leave['name']}**\n\n{leave['reason']}")
                        cols[1].write(f"{leave['start_date']} to {leave['end_date']}")
                        cols[2].write(f"Status: **{leave['status']}**")

                        if leave["status"] == "Pending":
                            subcols = cols[3].columns(2)
                            if subcols[0].button("Approve", key=f"app_{leave['id']}"):
                                update_leave_status(leave["id"], "Approved")
                                st.rerun()
                            if subcols[1].button("Reject", key=f"rej_{leave['id']}"):
                                update_leave_status(leave["id"], "Rejected")
                                st.rerun()
                        st.divider()
            else:
                st.info("No leave requests found.")
