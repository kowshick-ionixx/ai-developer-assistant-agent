import pytest
from calculator import add, subtract, multiply, divide, power, square_root

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 2) == 3
    assert subtract(2, 5) == -3

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6

def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ValueError, match="Division by zero"):
        divide(5, 0)

def test_power():
    assert power(2, 3) == 8
    assert power(9, 0.5) == 3

def test_square_root():
    assert square_root(9) == 3
    with pytest.raises(ValueError, match="Square root of negative"):
        square_root(-1)
