import pytest

from calculator import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0.5, 1.5) == 2.0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2
    assert subtract(5.5, 2.0) == 3.5


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0


def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(5, 2) == 2.5
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        divide(5, 0)
