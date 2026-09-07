import math


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


def power(a: float, b: float) -> float:
    return a**b


def square_root(a: float) -> float:
    if a < 0:
        raise ValueError("Square root of negative number is not supported.")
    return math.sqrt(a)
