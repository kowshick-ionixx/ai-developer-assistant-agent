import streamlit as st
import pandas as pd
from datetime import datetime
from database import ExpenseDatabase

st.set_page_config(page_title="Expense Tracker", page_icon="💰", layout="wide")

@st.cache_resource
def get_db():
    return ExpenseDatabase()

db = get_db()

st.title("💰 Personal Expense Tracker")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigation", ["Dashboard", "Manage Expenses"])

if menu == "Dashboard":
    st.header("Financial Dashboard")
    
    expenses = db.get_expenses()
    
    if not expenses:
        st.info("No expenses recorded yet. Go to 'Manage Expenses' to add some!")
    else:
        df = pd.DataFrame(expenses)
        df['amount'] = pd.to_numeric(df['amount'])
        
        total_spent = df['amount'].sum()
        total_transactions = len(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Expenses", f"${total_spent:,.2f}")
        with col2:
            st.metric("Total Transactions", total_transactions)
            
        st.subheader("Category-wise Spending")
        category_totals = db.get_category_totals()
        if category_totals:
            cat_df = pd.DataFrame(category_totals)
            cat_df.set_index('category', inplace=True)
            st.bar_chart(cat_df['total'])
            
        st.subheader("Recent Transactions")
        st.dataframe(df[['date', 'category', 'amount', 'description']].head(10), use_container_width=True)

elif menu == "Manage Expenses":
    st.header("Manage Expenses")
    
    with st.form("expense_form", clear_on_submit=True):
        st.subheader("Add New Expense")
        amount = st.number_input("Amount ($)", min_value=0.01, format="%.2f")
        category = st.selectbox("Category", ["Food", "Transport", "Housing", "Utilities", "Entertainment", "Other"])
        date = st.date_input("Date", value=datetime.today())
        description = st.text_input("Description")
        
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            if amount > 0:
                db.add_expense(amount, category, str(date), description)
                st.success("Expense added successfully!")
                st.rerun()
            else:
                st.error("Amount must be greater than zero.")
                
    st.subheader("Existing Expenses")
    expenses = db.get_expenses()
    
    if expenses:
        for exp in expenses:
            with st.expander(f"{exp['date']} | {exp['category']} | ${exp['amount']:.2f} - {exp['description'] or 'No description'}"):
                with st.form(f"edit_form_{exp['id']}"):
                    new_amount = st.number_input("Amount ($)", min_value=0.01, value=float(exp['amount']), format="%.2f", key=f"amt_{exp['id']}")
                    categories = ["Food", "Transport", "Housing", "Utilities", "Entertainment", "Other"]
                    default_cat_idx = categories.index(exp['category']) if exp['category'] in categories else 0
                    new_category = st.selectbox("Category", categories, index=default_cat_idx, key=f"cat_{exp['id']}")
                    new_date = st.date_input("Date", value=datetime.strptime(exp['date'], "%Y-%m-%d").date(), key=f"date_{exp['id']}")
                    new_desc = st.text_input("Description", value=exp['description'] or "", key=f"desc_{exp['id']}")
                    
                    col_save, col_del = st.columns(2)
                    with col_save:
                        if st.form_submit_button("Save Changes"):
                            db.update_expense(exp['id'], new_amount, new_category, str(new_date), new_desc)
                            st.success("Expense updated!")
                            st.rerun()
                    with col_del:
                        if st.form_submit_button("Delete"):
                            db.delete_expense(exp['id'])
                            st.success("Expense deleted!")
                            st.rerun()
    else:
        st.info("No expenses found.")
