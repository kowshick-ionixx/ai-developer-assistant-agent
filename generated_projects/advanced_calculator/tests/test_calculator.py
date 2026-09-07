import pytest

from calculator import (
    add,
    convert_length,
    convert_temperature,
    cosine,
    divide,
    factorial,
    multiply,
    power,
    sine,
    square_root,
    subtract,
    tangent,
)


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(4, 3) == 12
    assert multiply(-2, 3) == -6


def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ValueError):
        divide(5, 0)


def test_power():
    assert power(2, 3) == 8


def test_square_root():
    assert square_root(9) == 3.0
    with pytest.raises(ValueError):
        square_root(-1)


def test_factorial():
    assert factorial(5) == 120
    with pytest.raises(ValueError):
        factorial(-1)


def test_trig():
    assert sine(0) == 0.0
    assert cosine(0) == 1.0
    assert tangent(0) == 0.0


def test_convert_temperature():
    assert convert_temperature(0, "Celsius", "Celsius") == 0.0
    assert convert_temperature(0, "Celsius", "Fahrenheit") == 32.0
    assert convert_temperature(32, "Fahrenheit", "Celsius") == 0.0
    with pytest.raises(ValueError):
        convert_temperature(100, "Unknown", "Celsius")


def test_convert_length():
    assert convert_length(1, "Meters", "Centimeters") == 100.0
    assert convert_length(1000, "Meters", "Kilometers") == 1.0
    with pytest.raises(ValueError):
        convert_length(1, "Invalid", "Meters")
