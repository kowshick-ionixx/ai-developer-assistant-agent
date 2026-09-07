# Advanced Calculator

A complete, modern, professional Advanced Calculator application built using Python and Streamlit.

## Features
- **Basic Calculator**: Addition, subtraction, multiplication, division, modulus, exponentiation, square root, percentage, factorial, reciprocal, absolute value, sign change, decimal calculations, parentheses and operator precedence.
- **Scientific Calculator**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, log10, natural log (ln), e^x, 10^x, x², x³, xʸ, π, e, and Degrees/Radians mode.
- **Advanced Functions**: Combinations (nCr), Permutations (nPr), GCD, LCM, Floor, Ceiling, Scientific notation, Memory functions (MC, MR, M+, M-, MS), Calculation history, and Copy result.
- **Safety & Validation**: Safe mathematical parsing (no unsafe eval), robust error handling for division by zero, invalid inputs, negative square roots, negative factorials, invalid logarithms, and bad expressions.

## Installation & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```
