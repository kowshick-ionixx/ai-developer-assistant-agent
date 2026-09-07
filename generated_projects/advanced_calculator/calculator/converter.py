"""Unit conversion operations (Temperature and Length)."""

def convert_temperature(val: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return val
    
    if from_unit == "Celsius":
        c = val
    elif from_unit == "Fahrenheit":
        c = (val - 32) * 5/9
    elif from_unit == "Kelvin":
        c = val - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
        
    if to_unit == "Celsius":
        return c
    elif to_unit == "Fahrenheit":
        return c * 9/5 + 32
    elif to_unit == "Kelvin":
        return c + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")

def convert_length(val: float, from_unit: str, to_unit: str) -> float:
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
