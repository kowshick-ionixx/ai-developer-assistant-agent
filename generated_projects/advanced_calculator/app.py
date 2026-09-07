import streamlit as st
from calculator import (
    add, subtract, multiply, divide, power, square_root,
    factorial, sine, cosine, tangent, convert_temperature, convert_length
)

st.set_page_config(page_title="Advanced Calculator", page_icon="🧮", layout="centered")

st.title("🧮 Advanced Calculator App")
st.markdown("Perform basic arithmetic, advanced scientific calculations, and unit conversions.")

if "history" not in st.session_state:
    st.session_state.history = []

tab_basic, tab_scientific, tab_converter, tab_history = st.tabs([
    "Basic Arithmetic", "Scientific", "Unit Converter", "History"
])

with tab_basic:
    st.header("Basic Operations")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        num1 = st.number_input("First Number", value=0.0, format="%.4f", key="basic_num1")
    with col2:
        op = st.selectbox("Operation", ["+", "-", "*", "/", "^"])
    with col3:
        num2 = st.number_input("Second Number", value=0.0, format="%.4f", key="basic_num2")

    if st.button("Calculate", key="calc_basic"):
        try:
            if op == "+":
                res = add(num1, num2)
            elif op == "-":
                res = subtract(num1, num2)
            elif op == "*":
                res = multiply(num1, num2)
            elif op == "/":
                res = divide(num1, num2)
            elif op == "^":
                res = power(num1, num2)
            
            st.success(f"Result: {res}")
            st.session_state.history.append(f"{num1} {op} {num2} = {res}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab_scientific:
    st.header("Scientific Calculations")
    sci_op = st.selectbox("Function", ["Square Root", "Factorial", "Sine", "Cosine", "Tangent"])
    
    if sci_op in ["Sine", "Cosine", "Tangent"]:
        angle_unit = st.radio("Angle Unit", ["degrees", "radians"])
    else:
        angle_unit = "degrees"

    sci_num = st.number_input("Value", value=0.0, format="%.4f", key="sci_num")

    if st.button("Calculate", key="calc_sci"):
        try:
            if sci_op == "Square Root":
                res = square_root(sci_num)
                expr = f"sqrt({sci_num})"
            elif sci_op == "Factorial":
                res = factorial(int(sci_num))
                expr = f"factorial({int(sci_num)})"
            elif sci_op == "Sine":
                res = sine(sci_num, angle_unit)
                expr = f"sin({sci_num} {angle_unit})"
            elif sci_op == "Cosine":
                res = cosine(sci_num, angle_unit)
                expr = f"cos({sci_num} {angle_unit})"
            elif sci_op == "Tangent":
                res = tangent(sci_num, angle_unit)
                expr = f"tan({sci_num} {angle_unit})"

            st.success(f"Result: {res}")
            st.session_state.history.append(f"{expr} = {res}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab_converter:
    st.header("Unit Converter")
    conv_type = st.selectbox("Conversion Type", ["Temperature", "Length"])

    if conv_type == "Temperature":
        t_val = st.number_input("Temperature Value", value=0.0, format="%.2f", key="t_val")
        t_from = st.selectbox("From", ["Celsius", "Fahrenheit", "Kelvin"], key="t_from")
        t_to = st.selectbox("To", ["Celsius", "Fahrenheit", "Kelvin"], key="t_to")

        if st.button("Convert", key="conv_temp"):
            try:
                res = convert_temperature(t_val, t_from, t_to)
                st.success(f"Result: {res:.4f} {t_to}")
                st.session_state.history.append(f"{t_val} {t_from} -> {res:.4f} {t_to}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif conv_type == "Length":
        l_val = st.number_input("Length Value", value=0.0, format="%.4f", key="l_val")
        units = ["Meters", "Kilometers", "Centimeters", "Millimeters", "Miles", "Feet", "Inches"]
        l_from = st.selectbox("From", units, key="l_from")
        l_to = st.selectbox("To", units, key="l_to")

        if st.button("Convert", key="conv_len"):
            try:
                res = convert_length(l_val, l_from, l_to)
                st.success(f"Result: {res:.4f} {l_to}")
                st.session_state.history.append(f"{l_val} {l_from} -> {res:.4f} {l_to}")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_history:
    st.header("Calculation History")
    if not st.session_state.history:
        st.info("No calculations performed yet.")
    else:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
        for item in reversed(st.session_state.history):
            st.text(item)
