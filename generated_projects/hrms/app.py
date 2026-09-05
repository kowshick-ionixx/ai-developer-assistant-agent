import streamlit as st
import datetime
from database import (
    init_db,
    add_employee,
    get_employees,
    get_employee_by_id,
    mark_attendance,
    get_attendance,
    apply_leave,
    get_leaves,
    update_leave_status
)

st.set_page_config(page_title="Simple HRMS", page_icon="👥", layout="wide")

# Initialize DB
init_db()

st.sidebar.title("HRMS Menu")
menu = st.sidebar.selectbox("Navigate", ["Dashboard", "Employee Registration", "Employee List", "Attendance", "Leave Management"])

if menu == "Dashboard":
    st.title("📊 HRMS Dashboard")
    employees = get_employees()
    attendance = get_attendance()
    leaves = get_leaves()
    
    total_employees = len(employees)
    today = str(datetime.date.today())
    today_attendance = [a for a in attendance if a['date'] == today]
    present_today = len([a for a in today_attendance if a['status'] == 'Present'])
    pending_leaves = len([l for l in leaves if l['status'] == 'Pending'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", total_employees)
    col2.metric("Present Today", present_today)
    col3.metric("Pending Leaves", pending_leaves)
    
    st.subheader("Recent Employees")
    if employees:
        st.dataframe(employees[-5:])
    else:
        st.info("No employees registered yet.")

elif menu == "Employee Registration":
    st.title("📝 Employee Registration")
    
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        department = st.selectbox("Department", ["Engineering", "HR", "Sales", "Marketing", "Finance"])
        role = st.text_input("Role / Title")
        date_of_joining = st.date_input("Date of Joining", datetime.date.today())
        salary = st.number_input("Salary ($)", min_value=0.0, step=1000.0)
        
        submitted = st.form_submit_button("Register Employee")
        if submitted:
            if not name.strip() or not email.strip() or not role.strip():
                st.error("Please fill in all required fields.")
            else:
                success, msg = add_employee(name, email, department, role, str(date_of_joining), salary)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

elif menu == "Employee List":
    st.title("📋 Employee Directory & Details")
    employees = get_employees()
    
    if not employees:
        st.info("No employees found.")
    else:
        emp_names = {f"{e['id']}: {e['name']} ({e['email']})": e['id'] for e in employees}
        selected_option = st.selectbox("Select Employee", list(emp_names.keys()))
        selected_id = emp_names[selected_option]
        
        emp = get_employee_by_id(selected_id)
        if emp:
            st.subheader(f"Profile: {emp['name']}")
            col1, col2 = st.columns(2)
            col1.write(f"**Email:** {emp['email']}")
            col1.write(f"**Department:** {emp['department']}")
            col1.write(f"**Role:** {emp['role']}")
            col2.write(f"**Date of Joining:** {emp['date_of_joining']}")
            col2.write(f"**Salary:** ${emp['salary']:,.2f}")
            
            st.divider()
            st.subheader("Attendance History")
            all_att = get_attendance()
            emp_att = [a for a in all_att if a['employee_id'] == emp['id']]
            if emp_att:
                st.dataframe(emp_att)
            else:
                st.info("No attendance records for this employee.")
                
            st.subheader("Leave Requests")
            all_leaves = get_leaves()
            emp_leaves = [l for l in all_leaves if l['employee_id'] == emp['id']]
            if emp_leaves:
                st.dataframe(emp_leaves)
            else:
                st.info("No leave requests for this employee.")

elif menu == "Attendance":
    st.title("⏱️ Attendance Management")
    employees = get_employees()
    
    if not employees:
        st.warning("Please register employees first.")
    else:
        with st.form("attendance_form"):
            att_date = st.date_input("Date", datetime.date.today())
            emp_options = {f"{e['id']}: {e['name']}": e['id'] for e in employees}
            selected_emp = st.selectbox("Employee", list(emp_options.keys()))
            status = st.selectbox("Status", ["Present", "Absent", "Late"])
            
            submitted = st.form_submit_button("Record Attendance")
            if submitted:
                emp_id = emp_options[selected_emp]
                success, msg = mark_attendance(emp_id, str(att_date), status)
                st.success(msg)
                
        st.subheader("All Attendance Records")
        records = get_attendance()
        if records:
            st.dataframe(records)
        else:
            st.info("No records found.")

elif menu == "Leave Management":
    st.title("🏖️ Leave Management")
    employees = get_employees()
    
    if not employees:
        st.warning("Please register employees first.")
    else:
        tab1, tab2 = st.tabs(["Apply Leave", "Manage Leave Requests"])
        
        with tab1:
            with st.form("leave_form"):
                emp_options = {f"{e['id']}: {e['name']}": e['id'] for e in employees}
                selected_emp = st.selectbox("Employee", list(emp_options.keys()))
                start_date = st.date_input("Start Date", datetime.date.today())
                end_date = st.date_input("End Date", datetime.date.today())
                reason = st.text_area("Reason for Leave")
                
                submitted = st.form_submit_button("Submit Leave Application")
                if submitted:
                    if end_date < start_date:
                        st.error("End date cannot be earlier than start date.")
                    elif not reason.strip():
                        st.error("Please provide a reason.")
                    else:
                        emp_id = emp_options[selected_emp]
                        apply_leave(emp_id, str(start_date), str(end_date), reason)
                        st.success("Leave application submitted successfully.")
                        
        with tab2:
            leaves = get_leaves()
            if not leaves:
                st.info("No leave requests found.")
            else:
                for leave in leaves:
                    with st.expander(f"Leave #{leave['id']} - {leave['name']} ({leave['start_date']} to {leave['end_date']}) - Status: {leave['status']}"):
                        st.write(f"**Reason:** {leave['reason']}")
                        col1, col2 = st.columns(2)
                        if col1.button(f"Approve #{leave['id']}", key=f"app_{leave['id']}"):
                            update_leave_status(leave['id'], "Approved")
                            st.success("Approved!")
                            st.rerun()
                        if col2.button(f"Reject #{leave['id']}", key=f"rej_{leave['id']}"):
                            update_leave_status(leave['id'], "Rejected")
                            st.warning("Rejected!")
                            st.rerun()
