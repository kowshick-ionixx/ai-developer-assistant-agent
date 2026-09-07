import streamlit as st

from calculator import add, divide, multiply, subtract

st.set_page_config(page_title="Python Calculator", page_icon="🧮", layout="centered")

st.title("🧮 Python Calculator")
st.write(
    "A simple, clean calculator application supporting basic arithmetic operations."
)

# Initialize calculation history in session state
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)

with col1:
    num1 = st.number_input("First Number", value=0.0, format="%.4f")

with col2:
    num2 = st.number_input("Second Number", value=0.0, format="%.4f")

operation = st.selectbox(
    "Select Operation",
    ("Addition (+)", "Subtraction (-)", "Multiplication (×)", "Division (÷)"),
)

if st.button("Calculate", type="primary"):
    try:
        if operation == "Addition (+)":
            result = add(num1, num2)
            symbol = "+"
        elif operation == "Subtraction (-)":
            result = subtract(num1, num2)
            symbol = "-"
        elif operation == "Multiplication (×)":
            result = multiply(num1, num2)
            symbol = "×"
        elif operation == "Division (÷)":
            result = divide(num1, num2)
            symbol = "÷"

        expression = f"{num1} {symbol} {num2} = {result}"
        st.success(f"Result: **{result}**")
        st.session_state.history.insert(0, expression)
    except ValueError as e:
        st.error(f"Error: {e}")

if st.session_state.history:
    st.divider()
    st.subheader("Calculation History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    for item in st.session_state.history[:10]:
        st.text(item)
