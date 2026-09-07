"""Scientific calculator operations supporting degrees and radians."""

import math


def _to_radians(angle: float, mode: str) -> float:
    if mode.lower() == "deg":
        return math.radians(angle)
    return angle


def _from_radians(angle: float, mode: str) -> float:
    if mode.lower() == "deg":
        return math.degrees(angle)
    return angle


def sin_func(x: float, mode: str = "rad") -> float:
    return math.sin(_to_radians(x, mode))


def cos_func(x: float, mode: str = "rad") -> float:
    return math.cos(_to_radians(x, mode))


def tan_func(x: float, mode: str = "rad") -> float:
    rad = _to_radians(x, mode)
    cos_val = math.cos(rad)
    if abs(cos_val) < 1e-15:
        raise ValueError("Tangent undefined for this angle")
    return math.tan(rad)


def asin_func(x: float, mode: str = "rad") -> float:
    if x < -1 or x > 1:
        raise ValueError("Invalid input for arcsin (must be between -1 and 1)")
    return _from_radians(math.asin(x), mode)


def acos_func(x: float, mode: str = "rad") -> float:
    if x < -1 or x > 1:
        raise ValueError("Invalid input for arccos (must be between -1 and 1)")
    return _from_radians(math.acos(x), mode)


def atan_func(x: float, mode: str = "rad") -> float:
    return _from_radians(math.atan(x), mode)


def sinh_func(x: float) -> float:
    return math.sinh(x)


def cosh_func(x: float) -> float:
    return math.cosh(x)


def tanh_func(x: float) -> float:
    return math.tanh(x)


def log10_func(x: float) -> float:
    if x <= 0:
        raise ValueError("Invalid logarithm input (must be positive)")
    return math.log10(x)


def ln_func(x: float) -> float:
    if x <= 0:
        raise ValueError("Invalid natural logarithm input (must be positive)")
    return math.log(x)


def exp_func(x: float) -> float:
    return math.exp(x)


def power_10(x: float) -> float:
    return 10**x


def square(x: float) -> float:
    return x**2


def cube(x: float) -> float:
    return x**3


def power_xy(x: float, y: float) -> float:
    return x**y


PI = math.pi
E = math.e


# Aliases matching the names app.py imports.
# app.py passes a boolean is_degrees flag for the trig functions instead of
# the 'deg'/'rad' mode string the *_func implementations expect, so these
# need adapter wrappers rather than plain aliases.
def _mode_str(is_degrees: bool) -> str:
    return "deg" if is_degrees else "rad"


def sin_val(x: float, is_degrees: bool = False) -> float:
    return sin_func(x, _mode_str(is_degrees))


def cos_val(x: float, is_degrees: bool = False) -> float:
    return cos_func(x, _mode_str(is_degrees))


def tan_val(x: float, is_degrees: bool = False) -> float:
    return tan_func(x, _mode_str(is_degrees))


def asin_val(x: float, is_degrees: bool = False) -> float:
    return asin_func(x, _mode_str(is_degrees))


def acos_val(x: float, is_degrees: bool = False) -> float:
    return acos_func(x, _mode_str(is_degrees))


def atan_val(x: float, is_degrees: bool = False) -> float:
    return atan_func(x, _mode_str(is_degrees))


sinh_val = sinh_func
cosh_val = cosh_func
tanh_val = tanh_func
log10_val = log10_func
ln_val = ln_func
exp_val = exp_func
power10_val = power_10
power = power_xy


def pi_val() -> float:
    return PI


def e_val() -> float:
    return E
