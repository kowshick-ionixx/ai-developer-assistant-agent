import streamlit as st

from calculator import add, divide, multiply, power, square_root, subtract

st.set_page_config(page_title="Calculator App", page_icon="🧮", layout="centered")

st.title("🧮 Python Calculator App")
st.write("Perform basic and advanced calculations with a clean Streamlit interface.")

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = []

operation = st.selectbox(
    "Choose an operation",
    [
        "Addition (+)",
        "Subtraction (-)",
        "Multiplication (*)",
        "Division (/)",
        "Power (^)",
        "Square Root (√)",
    ],
)

try:
    if operation == "Square Root (√)":
        num = st.number_input("Enter number", value=0.0)
        if st.button("Calculate"):
            result = square_root(num)
            st.success(f"Result: √{num} = {result}")
            st.session_state.history.insert(0, f"√({num}) = {result}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            num1 = st.number_input("Enter first number", value=0.0)
        with col2:
            num2 = st.number_input("Enter second number", value=0.0)

        if st.button("Calculate"):
            if operation == "Addition (+)":
                result = add(num1, num2)
                symbol = "+"
            elif operation == "Subtraction (-)":
                result = subtract(num1, num2)
                symbol = "-"
            elif operation == "Multiplication (*)":
                result = multiply(num1, num2)
                symbol = "*"
            elif operation == "Division (/)":
                result = divide(num1, num2)
                symbol = "/"
            elif operation == "Power (^)":
                result = power(num1, num2)
                symbol = "^"

            st.success(f"Result: {num1} {symbol} {num2} = {result}")
            st.session_state.history.insert(0, f"{num1} {symbol} {num2} = {result}")

except Exception as e:
    st.error(f"Error: {e}")

# Display History
if st.session_state.history:
    st.subheader("📜 Calculation History")
    for item in st.session_state.history[:10]:
        st.text(item)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
