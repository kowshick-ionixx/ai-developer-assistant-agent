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
    return a ** b

def square_root(a: float) -> float:
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return math.sqrt(a)

def factorial(a: int) -> int:
    if not isinstance(a, int) or a < 0:
        raise ValueError("Factorial is only defined for non-negative integers.")
    return math.factorial(a)

def sine(a: float, unit: str = "degrees") -> float:
    val = math.radians(a) if unit == "degrees" else a
    return math.sin(val)

def cosine(a: float, unit: str = "degrees") -> float:
    val = math.radians(a) if unit == "degrees" else a
    return math.cos(val)

def tangent(a: float, unit: str = "degrees") -> float:
    val = math.radians(a) if unit == "degrees" else a
    return math.tan(val)

def convert_temperature(val: float, from_unit: str, to_unit: str) -> float:
    # Units: Celsius, Fahrenheit, Kelvin
    if from_unit == to_unit:
        return val
    
    # Convert to Celsius first
    if from_unit == "Celsius":
        c = val
    elif from_unit == "Fahrenheit":
        c = (val - 32) * 5/9
    elif from_unit == "Kelvin":
        c = val - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
        
    # Convert from Celsius to target
    if to_unit == "Celsius":
        return c
    elif to_unit == "Fahrenheit":
        return c * 9/5 + 32
    elif to_unit == "Kelvin":
        return c + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")

def convert_length(val: float, from_unit: str, to_unit: str) -> float:
    # Base unit: meters
    to_meters = {
        "Meters": 1.0,
        "Kilometers": 1000.0,
        "Centimeters": 0.01,
        "Millimeters": 0.001,
        "Miles": 1609.344,
        "Feet": 0.3048,
        "Inches": 0.0254
    }
    if from_unit not in to_meters or to_unit not in to_meters:
        raise ValueError("Invalid length unit")
        
    meters = val * to_meters[from_unit]
    return meters / to_meters[to_unit]
