import streamlit as st
from datetime import date, timedelta
from database import Database
from library import LibraryManager

st.set_page_config(page_title="Library Management System", page_icon="📚", layout="wide")

@st.cache_resource
def get_db():
    return Database()

db = get_db()
manager = LibraryManager(db)

st.title("📚 Library Management System")

tabs = st.tabs(["Dashboard", "Books", "Members", "Borrow & Return", "Reports"])

# 1. Dashboard Tab
with tabs[0]:
    st.header("Library Dashboard")
    stats = manager.get_dashboard_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Book Titles", stats["total_book_titles"])
    col2.metric("Total Copies", stats["total_books"])
    col3.metric("Available Copies", stats["available_books"])
    col4.metric("Active Loans", stats["active_loans"])
    col5.metric("Overdue Loans", stats["overdue_loans"], delta_color="inverse")

    st.markdown("---")
    st.subheader("Quick Overview")
    overdue_list = manager.get_overdue_borrowings()
    if overdue_list:
        st.warning(f"⚠️ There are {len(overdue_list)} overdue book(s)!")
        st.dataframe([{
            "Book": o['book_title'],
            "Member": o['member_name'],
            "Due Date": o['due_date'],
            "Days Overdue": o['days_overdue']
        } for o in overdue_list])
    else:
        st.success("✅ No overdue books at the moment.")

# 2. Books Tab
with tabs[1]:
    st.header("Books Management")
    
    sub_tabs = st.tabs(["View & Search Books", "Add Book", "Edit / Delete Book"])
    
    with sub_tabs[0]:
        col1, col2 = st.columns(2)
        search_query = col1.text_input("Search books (Title, Author, ISBN)", key="book_search")
        
        all_books = db.get_all_books()
        categories = ["All"] + sorted(list(set(b['category'] for b in all_books)))
        selected_cat = col2.selectbox("Filter by Category", categories)

        filtered = manager.search_books(search_query)
        if selected_cat != "All":
            filtered = [b for b in filtered if b['category'].lower() == selected_cat.lower()]

        if filtered:
            st.dataframe([{
                "ID": b['id'],
                "Title": b['title'],
                "Author": b['author'],
                "ISBN": b['isbn'],
                "Category": b['category'],
                "Total Qty": b['quantity'],
                "Available": b['available_quantity']
            } for b in filtered])
        else:
            st.info("No books found.")

    with sub_tabs[1]:
        st.subheader("Add New Book")
        with st.form("add_book_form"):
            title = st.text_input("Title")
            author = st.text_input("Author")
            isbn = st.text_input("ISBN")
            category = st.text_input("Category")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            submitted = st.form_submit_button("Add Book")
            if submitted:
                if not title or not author or not isbn or not category:
                    st.error("All fields are required.")
                else:
                    try:
                        db.add_book(title, author, isbn, category, quantity)
                        st.success(f"Book '{title}' added successfully!")
                    except Exception as e:
                        st.error(f"Error adding book: {e}")

    with sub_tabs[2]:
        st.subheader("Edit or Delete Book")
        all_books = db.get_all_books()
        if all_books:
            book_options = {f"{b['title']} (ISBN: {b['isbn']})": b['id'] for b in all_books}
            selected_book_label = st.selectbox("Select Book", list(book_options.keys()))
            selected_book_id = book_options[selected_book_label]
            book_data = db.get_book_by_id(selected_book_id)

            with st.form("edit_book_form"):
                e_title = st.text_input("Title", value=book_data['title'])
                e_author = st.text_input("Author", value=book_data['author'])
                e_isbn = st.text_input("ISBN", value=book_data['isbn'])
                e_category = st.text_input("Category", value=book_data['category'])
                e_qty = st.number_input("Total Quantity", min_value=0, value=book_data['quantity'], step=1)
                
                col_upd, col_del = st.columns(2)
                update_btn = col_upd.form_submit_button("Update Book")
                delete_btn = col_del.form_submit_button("Delete Book")

                if update_btn:
                    try:
                        db.update_book(selected_book_id, e_title, e_author, e_isbn, e_category, e_qty)
                        st.success("Book updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating book: {e}")

                if delete_btn:
                    try:
                        db.delete_book(selected_book_id)
                        st.success("Book deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting book: {e}")
        else:
            st.info("No books available to edit or delete.")

# 3. Members Tab
with tabs[2]:
    st.header("Members Management")
    sub_tabs = st.tabs(["View & Search Members", "Register Member", "Edit / Delete Member"])

    with sub_tabs[0]:
        search_query = st.text_input("Search members (ID, Name, Email)", key="member_search")
        filtered_members = manager.search_members(search_query)
        if filtered_members:
            st.dataframe([{
                "Database ID": m['id'],
                "Member ID": m['member_id'],
                "Name": m['name'],
                "Email": m['email'],
                "Phone": m['phone']
            } for m in filtered_members])
        else:
            st.info("No members found.")

    with sub_tabs[1]:
        st.subheader("Register New Member")
        with st.form("add_member_form"):
            m_id = st.text_input("Member ID / Card No")
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone Number")
            submitted = st.form_submit_button("Register Member")
            if submitted:
                if not m_id or not name or not email or not phone:
                    st.error("All fields are required.")
                else:
                    try:
                        db.add_member(m_id, name, email, phone)
                        st.success(f"Member '{name}' registered successfully!")
                    except Exception as e:
                        st.error(f"Error registering member: {e}")

    with sub_tabs[2]:
        st.subheader("Edit or Delete Member")
        all_members = db.get_all_members()
        if all_members:
            member_options = {f"{m['name']} ({m['member_id']})": m['id'] for m in all_members}
            selected_m_label = st.selectbox("Select Member", list(member_options.keys()))
            selected_m_id = member_options[selected_m_label]
            m_data = db.get_member_by_id(selected_m_id)

            with st.form("edit_member_form"):
                e_mid = st.text_input("Member ID", value=m_data['member_id'])
                e_name = st.text_input("Name", value=m_data['name'])
                e_email = st.text_input("Email", value=m_data['email'])
                e_phone = st.text_input("Phone", value=m_data['phone'])

                col_upd, col_del = st.columns(2)
                update_btn = col_upd.form_submit_button("Update Member")
                delete_btn = col_del.form_submit_button("Delete Member")

                if update_btn:
                    try:
                        db.update_member(selected_m_id, e_mid, e_name, e_email, e_phone)
                        st.success("Member updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating member: {e}")

                if delete_btn:
                    try:
                        db.delete_member(selected_m_id)
                        st.success("Member deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting member: {e}")
        else:
            st.info("No members available.")

# 4. Borrow & Return Tab
with tabs[3]:
    st.header("Borrowing & Returning Operations")
    b_sub_tabs = st.tabs(["Borrow a Book", "Return a Book"])

    with b_sub_tabs[0]:
        st.subheader("Issue Book Loan")
        available_books = [b for b in db.get_all_books() if b['available_quantity'] > 0]
        all_members = db.get_all_members()

        if not available_books:
            st.warning("No books currently available for borrowing.")
        elif not all_members:
            st.warning("No registered members available.")
        else:
            with st.form("borrow_form"):
                book_opts = {f"{b['title']} (Available: {b['available_quantity']})": b['id'] for b in available_books}
                selected_book_label = st.selectbox("Select Book", list(book_opts.keys()))
                book_id = book_opts[selected_book_label]

                member_opts = {f"{m['name']} ({m['member_id']})": m['id'] for m in all_members}
                selected_member_label = st.selectbox("Select Member", list(member_opts.keys()))
                member_id = member_opts[selected_member_label]

                borrow_date = st.date_input("Borrow Date", value=date.today())
                due_date = st.date_input("Due Date", value=date.today() + timedelta(days=14))

                borrow_submit = st.form_submit_button("Confirm Borrowing")
                if borrow_submit:
                    if due_date < borrow_date:
                        st.error("Due date cannot be before borrow date.")
                    else:
                        try:
                            db.borrow_book(book_id, member_id, borrow_date.strftime("%Y-%m-%d"), due_date.strftime("%Y-%m-%d"))
                            st.success("Book borrowed successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error borrowing book: {e}")

    with b_sub_tabs[1]:
        st.subheader("Return Borrowed Book")
        active_borrowings = [b for b in db.get_all_borrowings() if b['status'] == 'active']
        if active_borrowings:
            borrow_opts = {f"Book: {b['book_title']} | Member: {b['member_name']} (Due: {b['due_date']})": b['id'] for b in active_borrowings}
            selected_borrow_label = st.selectbox("Select Active Loan", list(borrow_opts.keys()))
            borrowing_id = borrow_opts[selected_borrow_label]
            return_date = st.date_input("Return Date", value=date.today())

            if st.button("Confirm Return"):
                try:
                    db.return_book(borrowing_id, return_date.strftime("%Y-%m-%d"))
                    st.success("Book returned successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error returning book: {e}")
        else:
            st.info("No active book loans to return.")

# 5. Reports Tab
with tabs[4]:
    st.header("Library Reports & Records")
    all_borrowings = db.get_all_borrowings()
    
    if all_borrowings:
        filter_status = st.selectbox("Filter Records by Status", ["All", "Active", "Returned", "Overdue"])
        
        display_records = []
        current_date = date.today()
        for b in all_borrowings:
            is_overdue = False
            if b['status'] == 'active':
                due = datetime.strptime(b['due_date'], "%Y-%m-%d").date()
                if due < current_date:
                    is_overdue = True

            status_label = "Active"
            if b['status'] == 'returned':
                status_label = "Returned"
            elif is_overdue:
                status_label = "Overdue"

            if filter_status == "Active" and b['status'] != 'active':
                continue
            if filter_status == "Returned" and b['status'] != 'returned':
                continue
            if filter_status == "Overdue" and not is_overdue:
                continue

            display_records.append({
                "Loan ID": b['id'],
                "Book": b['book_title'],
                "Member": b['member_name'],
                "Borrow Date": b['borrow_date'],
                "Due Date": b['due_date'],
                "Return Date": b['return_date'] or "Not Returned",
                "Status": status_label
            })

        st.dataframe(display_records)
    else:
        st.info("No borrowing records found.")
