import streamlit as st
import pandas as pd
from datetime import datetime
from database import init_db, add_transaction, get_transactions, delete_transaction

# Initialize database
init_db()

st.set_page_config(page_title="Expense Tracker App", page_icon="💰", layout="wide")

st.title("💰 Personal Expense Tracker")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Add Transaction", "History & Delete"])

if menu == "Dashboard":
    st.header("Financial Dashboard")
    df = get_transactions()
    
    if df.empty:
        st.info("No transactions recorded yet. Go to 'Add Transaction' to get started!")
    else:
        total_income = df[df['type'] == 'Income']['amount'].sum()
        total_expense = df[df['type'] == 'Expense']['amount'].sum()
        net_savings = total_income - total_expense
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Income", f"${total_income:,.2f}")
        col2.metric("Total Expenses", f"${total_expense:,.2f}")
        col3.metric("Net Savings", f"${net_savings:,.2f}")
        
        st.subheader("Recent Transactions")
        st.dataframe(df.head(10), use_container_width=True)
        
        if not df[df['type'] == 'Expense'].empty:
            st.subheader("Expenses by Category")
            exp_df = df[df['type'] == 'Expense']
            cat_summary = exp_df.groupby('category')['amount'].sum()
            st.bar_chart(cat_summary)

elif menu == "Add Transaction":
    st.header("Add New Transaction")
    with st.form("transaction_form"):
        t_type = st.selectbox("Type", ["Expense", "Income"])
        title = st.text_input("Description / Title")
        amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
        category = st.selectbox("Category", ["Food", "Rent", "Utilities", "Entertainment", "Transport", "Salary", "Freelance", "Other"])
        date = st.date_input("Date", value=datetime.today())
        
        submitted = st.form_submit_button("Save Transaction")
        if submitted:
            if title.strip() == "":
                st.error("Please enter a valid description.")
            else:
                add_transaction(str(date), title, amount, category, t_type)
                st.success(f"Added {t_type}: {title} (${amount:,.2f}) successfully!")

elif menu == "History & Delete":
    st.header("Transaction History & Management")
    df = get_transactions()
    
    if df.empty:
        st.info("No transactions found.")
    else:
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Delete a Transaction")
        t_id = st.selectbox("Select Transaction ID to Delete", df['id'].tolist())
        if st.button("Delete Selected"):
            delete_transaction(t_id)
            st.success(f"Transaction ID {t_id} deleted successfully!")
            st.experimental_rerun()
