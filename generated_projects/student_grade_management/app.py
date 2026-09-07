import pandas as pd
import streamlit as st
from database import (
    add_mark,
    add_student,
    calculate_grade,
    get_all_students,
    get_marks_for_student,
    init_db,
)

st.set_page_config(
    page_title="Student Grade Management", page_icon="🎓", layout="wide"
)

init_db()


def main():
    st.title("🎓 Student Grade Management System")

    menu = ["Add Student", "Enter Marks", "View Records"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Add Student":
        st.header("Add New Student")
        with st.form("student_form"):
            name = st.text_input("Full Name")
            roll_number = st.text_input("Roll Number / ID")
            class_name = st.text_input("Class / Grade")
            submitted = st.form_submit_button("Save Student")

            if submitted:
                if name and roll_number and class_name:
                    success, msg = add_student(name, roll_number, class_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill out all fields.")

    elif choice == "Enter Marks":
        st.header("Enter Subject Marks")
        students = get_all_students()
        if not students:
            st.warning("Please add students first.")
        else:
            student_dict = {f"{s[1]} ({s[2]})": s[0] for s in students}
            selected_student_label = st.selectbox(
                "Select Student", list(student_dict.keys())
            )
            student_id = student_dict[selected_student_label]

            with st.form("marks_form"):
                subject = st.text_input("Subject Name")
                mark = st.number_input(
                    "Mark Obtained", min_value=0.0, max_value=100.0, step=0.5
                )
                submitted = st.form_submit_button("Save Mark")

                if submitted:
                    if subject:
                        add_mark(student_id, subject, mark)
                        st.success(f"Added mark for {subject} successfully.")
                    else:
                        st.warning("Please enter a subject name.")

    elif choice == "View Records":
        st.header("Student Records & Summary")
        students = get_all_students()
        if not students:
            st.info("No student records found.")
        else:
            summary_data = []
            for s in students:
                s_id, name, roll, cls = s
                marks = get_marks_for_student(s_id)
                if marks:
                    total = sum(m[1] for m in marks)
                    avg = total / len(marks)
                    grade = calculate_grade(avg)
                else:
                    total = 0.0
                    avg = 0.0
                    grade = "N/A"

                summary_data.append(
                    {
                        "Roll Number": roll,
                        "Name": name,
                        "Class": cls,
                        "Total Marks": total,
                        "Average": round(avg, 2),
                        "Grade": grade,
                    }
                )

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)

            st.subheader("Detailed Marks Breakdown")
            selected_student_detail = st.selectbox(
                "Select Student to View Marks",
                list({f"{s[1]} ({s[2]})": s[0] for s in students}.keys()),
                key="detail_view",
            )
            detail_id = {f"{s[1]} ({s[2]})": s[0] for s in students}[
                selected_student_detail
            ]
            student_marks = get_marks_for_student(detail_id)
            if student_marks:
                df_marks = pd.DataFrame(student_marks, columns=["Subject", "Mark"])
                st.table(df_marks)
            else:
                st.info("No marks entered for this student yet.")


if __name__ == "__main__":
    main()
