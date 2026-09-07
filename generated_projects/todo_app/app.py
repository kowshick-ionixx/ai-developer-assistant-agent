import streamlit as st

st.set_page_config(page_title="To-Do List App", page_icon="✅", layout="centered")

def init_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = []

def add_task(title, description=""):
    if title.strip():
        st.session_state.tasks.append({
            "title": title.strip(),
            "description": description.strip(),
            "completed": False
        })
        return True
    return False

def toggle_task(index):
    if 0 <= index < len(st.session_state.tasks):
        st.session_state.tasks[index]["completed"] = not st.session_state.tasks[index]["completed"]

def delete_task(index):
    if 0 <= index < len(st.session_state.tasks):
        st.session_state.tasks.pop(index)

def main():
    init_state()
    
    st.title("✅ Python To-Do List App")
    st.markdown("A simple and clean task management app built with Streamlit.")

    # Sidebar for adding tasks
    st.sidebar.header("Add New Task")
    with st.sidebar.form("task_form", clear_on_submit=True):
        title = st.text_input("Task Title")
        description = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Add Task")
        if submitted:
            if add_task(title, description):
                st.sidebar.success(f"Added task: {title}")
                st.rerun()
            else:
                st.sidebar.error("Task title cannot be empty.")

    # Main area - Filter & Tasks
    filter_option = st.radio("Filter Tasks", ["All", "Active", "Completed"], horizontal=True)

    tasks = st.session_state.tasks

    if filter_option == "Active":
        filtered_indices = [i for i, t in enumerate(tasks) if not t["completed"]]
    elif filter_option == "Completed":
        filtered_indices = [i for i, t in enumerate(tasks) if t["completed"]]
    else:
        filtered_indices = list(range(len(tasks)))

    if not filtered_indices:
        st.info(f"No {filter_option.lower()} tasks found.")
    else:
        st.markdown(f"### Tasks ({len(filtered_indices)})")
        for i in filtered_indices:
            task = tasks[i]
            col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
            with col1:
                checked = st.checkbox("", value=task["completed"], key=f"chk_{i}")
                if checked != task["completed"]:
                    toggle_task(i)
                    st.rerun()
            with col2:
                if task["completed"]:
                    st.markdown(f"~~**{task['title']}**~~")
                else:
                    st.markdown(f"**{task['title']}**")
                if task["description"]:
                    st.caption(task["description"])
            with col3:
                if st.button("Delete", key=f"del_{i}"):
                    delete_task(i)
                    st.rerun()

    # Clear completed button
    completed_count = sum(1 for t in tasks if t["completed"])
    if completed_count > 0 and st.button("Clear Completed Tasks"):
        st.session_state.tasks = [t for t in tasks if not t["completed"]]
        st.rerun()

if __name__ == "__main__":
    main()
