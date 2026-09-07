import math

class AdvancedCalculator:
    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def subtract(a: float, b: float) -> float:
        return a - b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    @staticmethod
    def power(base: float, exp: float) -> float:
        return base ** exp

    @staticmethod
    def square_root(a: float) -> float:
        if a < 0:
            raise ValueError("Square root of negative number is not supported in real domain.")
        return math.sqrt(a)

    @staticmethod
    def logarithm(a: float, base: float = math.e) -> float:
        if a <= 0:
            raise ValueError("Logarithm argument must be positive.")
        if base <= 0 or base == 1:
            raise ValueError("Invalid logarithm base.")
        return math.log(a, base)

    @staticmethod
    def sine(angle: float, degrees: bool = True) -> float:
        rad = math.radians(angle) if degrees else angle
        return math.sin(rad)

    @staticmethod
    def cosine(angle: float, degrees: bool = True) -> float:
        rad = math.radians(angle) if degrees else angle
        return math.cos(rad)

    @staticmethod
    def tangent(angle: float, degrees: bool = True) -> float:
        rad = math.radians(angle) if degrees else angle
        return math.tan(rad)

    @staticmethod
    def factorial(n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("Factorial is only defined for non-negative integers.")
        return math.factorial(n)

    @staticmethod
    def convert_units(value: float, category: str, from_unit: str, to_unit: str) -> float:
        length_to_m = {
            "meters": 1.0,
            "kilometers": 1000.0,
            "centimeters": 0.01,
            "millimeters": 0.001,
            "miles": 1609.34,
            "feet": 0.3048,
            "inches": 0.0254
        }
        
        if category == "Temperature":
            if from_unit == to_unit:
                return value
            if from_unit == "Celsius":
                c = value
            elif from_unit == "Fahrenheit":
                c = (value - 32) * 5/9
            elif from_unit == "Kelvin":
                c = value - 273.15
            else:
                raise ValueError(f"Unknown unit: {from_unit}")
            
            if to_unit == "Celsius":
                return c
            elif to_unit == "Fahrenheit":
                return c * 9/5 + 32
            elif to_unit == "Kelvin":
                return c + 273.15
            else:
                raise ValueError(f"Unknown unit: {to_unit}")

        if category == "Length":
            if from_unit not in length_to_m or to_unit not in length_to_m:
                raise ValueError("Invalid length unit.")
            meters = value * length_to_m[from_unit]
            return meters / length_to_m[to_unit]

        raise ValueError(f"Unknown category: {category}")
