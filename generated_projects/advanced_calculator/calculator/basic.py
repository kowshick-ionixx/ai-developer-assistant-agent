"""Basic calculator operations."""

import math


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


def modulus(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Modulo by zero")
    return a % b


def power(a: float, b: float) -> float:
    try:
        return a**b
    except Exception as e:
        raise ValueError(f"Invalid exponentiation: {e}")


exponent = power


def square_root(a: float) -> float:
    if a < 0:
        raise ValueError("Negative square root")
    return math.sqrt(a)


def percentage(a: float, b: float = 100.0) -> float:
    return (a * b) / 100.0


def factorial(n: float) -> int:
    if not isinstance(n, (int, float)) or n < 0 or not math.isfinite(n) or n != int(n):
        raise ValueError("Invalid factorial input")
    if n > 170:
        raise ValueError("Factorial result too large")
    return math.factorial(int(n))


def reciprocal(a: float) -> float:
    if a == 0:
        raise ValueError("Division by zero in reciprocal")
    return 1.0 / a


def absolute_value(a: float) -> float:
    return abs(a)


absolute = absolute_value


def sign_change(a: float) -> float:
    return -a
