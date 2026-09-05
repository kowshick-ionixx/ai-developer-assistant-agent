import streamlit as st
from datetime import datetime
from database import ExpenseDatabase

st.set_page_config(page_title="Expense Tracker", page_icon="💰", layout="wide")

@st.cache_resource
def get_db():
    return ExpenseDatabase()

db = get_db()

st.title("💰 Simple Expense Tracker")

menu = ["Dashboard", "Add Expense", "Manage Expenses"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Dashboard":
    st.header("Dashboard & Summary")
    
    total = db.get_total_expense()
    st.metric(label="Total Expenses", value=f"${total:,.2f}")
    
    st.subheader("Category Summary")
    summary = db.get_category_summary()
    if summary:
        col1, col2 = st.columns(2)
        with col1:
            st.table([{"Category": row[0], "Total Amount ($)": f"{row[1]:,.2f}"} for row in summary])
        with col2:
            chart_data = {row[0]: row[1] for row in summary}
            st.bar_chart(chart_data)
    else:
        st.info("No expenses recorded yet. Add some from the 'Add Expense' tab!")

elif choice == "Add Expense":
    st.header("Add New Expense")
    
    with st.form("add_expense_form"):
        categories = ["Food", "Housing", "Transport", "Utilities", "Entertainment", "Healthcare", "Other"]
        category = st.selectbox("Category", categories)
        amount = st.number_input("Amount ($)", min_value=0.01, step=1.00, format="%.2f")
        date = st.date_input("Date", value=datetime.today())
        description = st.text_input("Description (Optional)")
        
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            try:
                db.add_expense(category, amount, date.strftime("%Y-%m-%d"), description)
                st.success("Expense added successfully!")
            except ValueError as e:
                st.error(f"Error: {e}")

elif choice == "Manage Expenses":
    st.header("View, Edit & Delete Expenses")
    
    expenses = db.get_expenses()
    if not expenses:
        st.info("No expenses found.")
    else:
        for exp in expenses:
            exp_id, cat, amt, dt, desc = exp
            with st.expander(f"{dt} | {cat} | ${amt:,.2f} {f'- {desc}' if desc else ''}"):
                with st.form(f"edit_form_{exp_id}"):
                    categories = ["Food", "Housing", "Transport", "Utilities", "Entertainment", "Healthcare", "Other"]
                    new_cat = st.selectbox("Category", categories, index=categories.index(cat) if cat in categories else 0)
                    new_amt = st.number_input("Amount ($)", min_value=0.01, value=float(amt), step=1.00, format="%.2f", key=f"amt_{exp_id}")
                    try:
                        default_date = datetime.strptime(dt, "%Y-%m-%d").date()
                    except ValueError:
                        default_date = datetime.today().date()
                    new_dt = st.date_input("Date", value=default_date, key=f"dt_{exp_id}")
                    new_desc = st.text_input("Description", value=desc, key=f"desc_{exp_id}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        update_submitted = st.form_submit_button("Update Expense")
                    with col2:
                        delete_submitted = st.form_submit_button("Delete Expense")
                        
                    if update_submitted:
                        try:
                            db.update_expense(exp_id, new_cat, new_amt, new_dt.strftime("%Y-%m-%d"), new_desc)
                            st.success("Expense updated successfully!")
                            st.rerun()
                        except ValueError as e:
                            st.error(f"Error: {e}")
                    
                    if delete_submitted:
                        db.delete_expense(exp_id)
                        st.success("Expense deleted successfully!")
                        st.rerun()
