import pytest
from calculator.basic import (
    absolute_value,
    add,
    divide,
    factorial,
    modulus,
    multiply,
    percentage,
    power,
    reciprocal,
    sign_change,
    square_root,
    subtract,
)


def test_basic_arithmetic():
    assert add(10, 20) == 30
    assert subtract(50, 20) == 30
    assert multiply(6, 7) == 42
    assert divide(100, 4) == 25
    assert modulus(10, 3) == 1
    assert power(2, 10) == 1024


def test_basic_functions():
    assert square_root(144) == 12
    assert percentage(50, 200) == 100.0
    assert factorial(5) == 120
    assert reciprocal(4) == 0.25
    assert absolute_value(-15) == 15
    assert sign_change(10) == -10


def test_basic_errors():
    with pytest.raises(ValueError, match="Division by zero"):
        divide(10, 0)
    with pytest.raises(ValueError, match="Negative square root"):
        square_root(-9)
    with pytest.raises(ValueError, match="Invalid factorial input"):
        factorial(-1)
    with pytest.raises(ValueError, match="Division by zero in reciprocal"):
        reciprocal(0)
