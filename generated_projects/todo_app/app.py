import streamlit as st
from database import init_db, add_task, get_tasks, complete_task, delete_task, get_task_counts

def main():
    st.set_page_config(page_title="Simple To-Do List Application", page_icon="📝", layout="centered")
    init_db()

    st.title("📝 Simple To-Do List Application")

    # Add task form
    with st.form("add_task_form", clear_on_submit=True):
        task_desc = st.text_input("New Task Description")
        submitted = st.form_submit_button("Add Task")
        if submitted:
            if not task_desc or not task_desc.strip():
                st.error("Task description cannot be empty.")
            else:
                try:
                    add_task(task_desc)
                    st.success("Task added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding task: {e}")

    st.markdown("---")

    # Display counts
    counts = get_task_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", counts["total"])
    col2.metric("Completed", counts["completed"])
    col3.metric("Pending", counts["pending"])

    st.markdown("---")

    # View tasks
    st.subheader("Your Tasks")
    tasks = get_tasks()

    if not tasks:
        st.info("No tasks yet. Add one above!")
        return

    for task in tasks:
        cols = st.columns([0.6, 0.2, 0.2])
        status_str = "✅" if task["completed"] else "⏳"
        desc_text = f"~~{task['description']}~~" if task["completed"] else task["description"]
        
        cols[0].write(f"{status_str} {desc_text} *({task['created_date']})*")
        
        if not task["completed"]:
            if cols[1].button("Complete", key=f"complete_{task['id']}"):
                complete_task(task['id'])
                st.rerun()
        else:
            cols[1].write("")

        if cols[2].button("Delete", key=f"delete_{task['id']}"):
            delete_task(task['id'])
            st.rerun()

if __name__ == "__main__":
    main()
