import streamlit as st
from database import init_db, add_task, get_tasks, complete_task, delete_task

def main():
    st.set_page_config(page_title="To-Do App", page_icon="✅", layout="centered")
    
    # Initialize DB
    init_db()

    st.title("📝 Simple To-Do Application")

    # Input section
    st.subheader("Add a New Task")
    with st.form("task_form", clear_on_submit=True):
        task_title = st.text_input("Task Title")
        submitted = st.form_submit_button("Add Task")
        if submitted:
            if not task_title or not task_title.strip():
                st.error("Task title cannot be empty!")
            else:
                try:
                    add_task(task_title)
                    st.success(f"Added task: {task_title}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding task: {e}")

    # Display tasks
    tasks = get_tasks()
    pending_tasks = [t for t in tasks if t["status"] == "pending"]
    completed_tasks = [t for t in tasks if t["status"] == "completed"]

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📌 Pending Tasks ({len(pending_tasks)})")
        if not pending_tasks:
            st.info("No pending tasks.")
        else:
            for task in pending_tasks:
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.text(task["title"])
                if c2.button("✔", key=f"comp_{task['id']}"):
                    complete_task(task["id"])
                    st.rerun()
                if c3.button("❌", key=f"del_{task['id']}"):
                    delete_task(task["id"])
                    st.rerun()

    with col2:
        st.subheader(f"✅ Completed Tasks ({len(completed_tasks)})")
        if not completed_tasks:
            st.info("No completed tasks.")
        else:
            for task in completed_tasks:
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"~~{task['title']}~~")
                if c2.button("🗑️", key=f"del_comp_{task['id']}"):
                    delete_task(task["id"])
                    st.rerun()

if __name__ == "__main__":
    main()
