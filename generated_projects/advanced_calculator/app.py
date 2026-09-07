import streamlit as st
import math
from calculator.basic import add, subtract, multiply, divide, modulus, exponent, square_root, percentage, factorial, reciprocal, absolute, sign_change
from calculator.scientific import sin_val, cos_val, tan_val, asin_val, acos_val, atan_val, sinh_val, cosh_val, tanh_val, log10_val, ln_val, exp_val, power10_val, square, cube, power, pi_val, e_val
from calculator.advanced import nCr, nPr, gcd_val, lcm_val, floor_val, ceil_val, sci_notation
from calculator.parser import evaluate_expression
from calculator.memory import Memory

st.set_page_config(page_title="Advanced Calculator", page_icon="🧮", layout="centered")

if "memory" not in st.session_state:
    st.session_state.memory = Memory()
if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = "0"
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "Degrees"

st.title("🧮 Advanced Calculator")
st.markdown("A modern, clean, professional calculator with Basic, Scientific, and Advanced modes.")

# Sidebar for Memory and History
with st.sidebar:
    st.header("🧠 Memory & History")
    
    # Memory controls
    st.subheader("Memory Functions")
    mem_val = st.session_state.memory.recall()
    st.info(f"Memory Value: {mem_val}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    if col_m1.button("MC"):
        st.session_state.memory.clear()
        st.rerun()
    if col_m2.button("MR"):
        st.session_state.display = str(st.session_state.memory.recall())
        st.rerun()
    if col_m3.button("MS"):
        try:
            val = float(st.session_state.display)
            st.session_state.memory.store(val)
            st.success(f"Stored {val}")
        except Exception:
            st.error("Invalid value to store")
            
    col_m4, col_m5 = st.columns(2)
    if col_m4.button("M+"):
        try:
            val = float(st.session_state.display)
            curr = st.session_state.memory.recall()
            st.session_state.memory.store(curr + val)
            st.success(f"Added {val}")
        except Exception:
            st.error("Invalid value")
    if col_m5.button("M-"):
        try:
            val = float(st.session_state.display)
            curr = st.session_state.memory.recall()
            st.session_state.memory.store(curr - val)
            st.success(f"Subtracted {val}")
        except Exception:
            st.error("Invalid value")

    st.markdown("---")
    st.subheader("Calculation History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
        
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-20:])):
            st.text(item)
    else:
        st.text("No history yet.")

# Main calculator layout
mode = st.selectbox("Calculator Mode", ["Basic Calculator", "Scientific Calculator", "Advanced Functions", "Expression Parser"])

# Angle mode toggle for trigonometric functions
angle_mode = st.radio("Angle Mode", ["Degrees", "Radians"], horizontal=True)
st.session_state.angle_mode = angle_mode
is_degrees = (angle_mode == "Degrees")

# Display screen
st.markdown(f"### Current Display: `{st.session_state.display}`")

if mode == "Basic Calculator":
    st.subheader("Basic Operations")
    
    # Quick Expression Input
    expr_input = st.text_input("Enter expression or use buttons below:", value=st.session_state.display)
    if st.button("Calculate Expression"):
        try:
            res = evaluate_expression(expr_input)
            st.session_state.display = str(res)
            st.session_state.history.append(f"{expr_input} = {res}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("C"):
        st.session_state.display = "0"
        st.rerun()
    if col2.button("+/-"):
        try:
            res = sign_change(float(st.session_state.display))
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col3.button("%"):
        try:
            res = percentage(float(st.session_state.display))
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col4.button("÷"):
        st.session_state.display += " / "
        st.rerun()

    col5, col6, col7, col8 = st.columns(4)
    if col5.button("7"):
        st.session_state.display = st.session_state.display + "7" if st.session_state.display != "0" else "7"
        st.rerun()
    if col6.button("8"):
        st.session_state.display = st.session_state.display + "8" if st.session_state.display != "0" else "8"
        st.rerun()
    if col7.button("9"):
        st.session_state.display = st.session_state.display + "9" if st.session_state.display != "0" else "9"
        st.rerun()
    if col8.button("×"):
        st.session_state.display += " * "
        st.rerun()

    col9, col10, col11, col12 = st.columns(4)
    if col9.button("4"):
        st.session_state.display = st.session_state.display + "4" if st.session_state.display != "0" else "4"
        st.rerun()
    if col10.button("5"):
        st.session_state.display = st.session_state.display + "5" if st.session_state.display != "0" else "5"
        st.rerun()
    if col11.button("6"):
        st.session_state.display = st.session_state.display + "6" if st.session_state.display != "0" else "6"
        st.rerun()
    if col12.button("-"):
        st.session_state.display += " - "
        st.rerun()

    col13, col14, col15, col16 = st.columns(4)
    if col13.button("1"):
        st.session_state.display = st.session_state.display + "1" if st.session_state.display != "0" else "1"
        st.rerun()
    if col14.button("2"):
        st.session_state.display = st.session_state.display + "2" if st.session_state.display != "0" else "2"
        st.rerun()
    if col15.button("3"):
        st.session_state.display = st.session_state.display + "3" if st.session_state.display != "0" else "3"
        st.rerun()
    if col16.button("+"):
        st.session_state.display += " + "
        st.rerun()

    col17, col18, col19, col20 = st.columns(4)
    if col17.button("0"):
        st.session_state.display = st.session_state.display + "0" if st.session_state.display != "0" else "0"
        st.rerun()
    if col18.button("."):
        if "." not in st.session_state.display.split()[-1]:
            st.session_state.display += "."
        st.rerun()
    if col19.button("1/x"):
        try:
            res = reciprocal(float(st.session_state.display))
            st.session_state.display = str(res)
            st.session_state.history.append(f"1/{st.session_state.display} = {res}")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col20.button("="):
        try:
            res = evaluate_expression(st.session_state.display)
            st.session_state.history.append(f"{st.session_state.display} = {res}")
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    col21, col22, col23, col24 = st.columns(4)
    if col21.button("x²"):
        try:
            res = square(float(st.session_state.display))
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col22.button("√x"):
        try:
            res = square_root(float(st.session_state.display))
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col23.button("xʸ"):
        st.session_state.display += " ** "
        st.rerun()
    if col24.button("n!"):
        try:
            res = factorial(int(float(st.session_state.display)))
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))

elif mode == "Scientific Calculator":
    st.subheader("Scientific Operations")
    
    try:
        val = float(st.session_state.display)
    except:
        val = 0.0

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("sin(x)"):
        res = sin_val(val, is_degrees)
        st.session_state.history.append(f"sin({val}) = {res}")
        st.session_state.display = str(res)
        st.rerun()
    if col2.button("cos(x)"):
        res = cos_val(val, is_degrees)
        st.session_state.history.append(f"cos({val}) = {res}")
        st.session_state.display = str(res)
        st.rerun()
    if col3.button("tan(x)"):
        try:
            res = tan_val(val, is_degrees)
            st.session_state.history.append(f"tan({val}) = {res}")
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col4.button("log10(x)"):
        try:
            res = log10_val(val)
            st.session_state.history.append(f"log10({val}) = {res}")
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))

    col5, col6, col7, col8 = st.columns(4)
    if col5.button("asin(x)"):
        try:
            res = asin_val(val, is_degrees)
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col6.button("acos(x)"):
        try:
            res = acos_val(val, is_degrees)
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if col7.button("atan(x)"):
        res = atan_val(val, is_degrees)
        st.session_state.display = str(res)
        st.rerun()
    if col8.button("ln(x)"):
        try:
            res = ln_val(val)
            st.session_state.display = str(res)
            st.rerun()
        except Exception as e:
            st.error(str(e))

    col9, col10, col11, col12 = st.columns(4)
    if col9.button("sinh(x)"):
        res = sinh_val(val)
        st.session_state.display = str(res)
        st.rerun()
    if col10.button("cosh(x)"):
        res = cosh_val(val)
        st.session_state.display = str(res)
        st.rerun()
    if col11.button("tanh(x)"):
        res = tanh_val(val)
        st.session_state.display = str(res)
        st.rerun()
    if col12.button("e^x"):
        res = exp_val(val)
        st.session_state.display = str(res)
        st.rerun()

    col13, col14, col15, col16 = st.columns(4)
    if col13.button("x²"):
        res = square(val)
        st.session_state.display = str(res)
        st.rerun()
    if col14.button("x³"):
        res = cube(val)
        st.session_state.display = str(res)
        st.rerun()
    if col15.button("10^x"):
        res = power10_val(val)
        st.session_state.display = str(res)
        st.rerun()
    if col16.button("π"):
        st.session_state.display = str(pi_val())
        st.rerun()
        
    if st.button("e"):
        st.session_state.display = str(e_val())
        st.rerun()

elif mode == "Advanced Functions":
    st.subheader("Advanced Number Theory & Combinatorics")
    
    tab_comb, tab_num = st.tabs(["Combinatorics", "Number Theory / Formatting"])
    
    with tab_comb:
        st.markdown("### Permutations & Combinations")
        n_val = st.number_input("n", min_value=0, value=5, step=1)
        r_val = st.number_input("r", min_value=0, value=2, step=1)
        
        col_c1, col_c2 = st.columns(2)
        if col_c1.button("Calculate nCr"):
            try:
                res = nCr(int(n_val), int(r_val))
                st.success(f"{n_val}C{r_val} = {res}")
                st.session_state.history.append(f"{n_val}C{r_val} = {res}")
            except Exception as e:
                st.error(str(e))
        if col_c2.button("Calculate nPr"):
            try:
                res = nPr(int(n_val), int(r_val))
                st.success(f"{n_val}P{r_val} = {res}")
                st.session_state.history.append(f"{n_val}P{r_val} = {res}")
            except Exception as e:
                st.error(str(e))
                
    with tab_num:
        st.markdown("### GCD, LCM, Floor, Ceiling, Scientific Notation")
        a_val = st.number_input("Value A / Number", value=48.5)
        b_val = st.number_input("Value B (for GCD/LCM)", value=18.0)
        
        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
        if col_n1.button("GCD"):
            try:
                res = gcd_val(int(a_val), int(b_val))
                st.success(f"GCD = {res}")
            except Exception as e:
                st.error(str(e))
        if col_n2.button("LCM"):
            try:
                res = lcm_val(int(a_val), int(b_val))
                st.success(f"LCM = {res}")
            except Exception as e:
                st.error(str(e))
        if col_n3.button("Floor"):
            st.success(f"Floor = {floor_val(a_val)}")
        if col_n4.button("Ceiling"):
            st.success(f"Ceiling = {ceil_val(a_val)}")
            
        if st.button("Scientific Notation"):
            st.success(f"Sci Notation = {sci_notation(a_val)}")

elif mode == "Expression Parser":
    st.subheader("Safe Expression Evaluator")
    st.markdown("Enter any valid mathematical expression (e.g., `(10 + 20) * 2 ** 3 / 4`):")
    custom_expr = st.text_input("Expression", value="10 + 20")
    if st.button("Evaluate"):
        try:
            res = evaluate_expression(custom_expr)
            st.success(f"Result: {res}")
            st.session_state.history.append(f"{custom_expr} = {res}")
            st.session_state.display = str(res)
        except Exception as e:
            st.error(f"Error evaluating expression: {e}")
