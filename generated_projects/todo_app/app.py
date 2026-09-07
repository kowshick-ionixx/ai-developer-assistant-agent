import sqlite3
import streamlit as st

DB_FILE = "todos.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT NOT NULL, completed INTEGER DEFAULT 0)"
    )
    conn.commit()
    conn.close()


def get_todos():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, task, completed FROM todos")
    todos = c.fetchall()
    conn.close()
    return todos


def add_todo(task):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO todos (task, completed) VALUES (?, 0)", (task,))
    conn.commit()
    conn.close()


def toggle_todo(todo_id, completed):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "UPDATE todos SET completed = ? WHERE id = ?", (1 if completed else 0, todo_id)
    )
    conn.commit()
    conn.close()


def delete_todo(todo_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    conn.commit()
    conn.close()


def main():
    st.title("Simple Todo App")
    init_db()

    new_task = st.text_input("Add a new task")
    if st.button("Add"):
        if new_task.strip():
            add_todo(new_task.strip())
            st.success(f"Added task: {new_task}")
            st.rerun()
        else:
            st.warning("Task cannot be empty.")

    st.subheader("Your Tasks")
    todos = get_todos()
    if not todos:
        st.info("No tasks yet. Add one above!")
    for todo_id, task, completed in todos:
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
        with col1:
            is_checked = st.checkbox(
                "", value=bool(completed), key=f"check_{todo_id}"
            )
            if is_checked != bool(completed):
                toggle_todo(todo_id, is_checked)
                st.rerun()
        with col2:
            if completed:
                st.markdown(f"~~{task}~~")
            else:
                st.write(task)
        with col3:
            if st.button("Delete", key=f"del_{todo_id}"):
                delete_todo(todo_id)
                st.rerun()


if __name__ == "__main__":
    main()
