import math

import pytest
from calculator.scientific import (
    E,
    cos_func,
    cube,
    exp_func,
    ln_func,
    log10_func,
    power_10,
    power_xy,
    sin_func,
    square,
    tan_func,
)


def test_trig_functions():
    assert math.isclose(sin_func(90, mode="deg"), 1.0, abs_tol=1e-9)
    assert math.isclose(cos_func(0, mode="deg"), 1.0, abs_tol=1e-9)
    assert math.isclose(tan_func(45, mode="deg"), 1.0, abs_tol=1e-9)


def test_log_and_exp():
    assert math.isclose(log10_func(100), 2.0, abs_tol=1e-9)
    assert math.isclose(ln_func(E), 1.0, abs_tol=1e-9)
    assert math.isclose(exp_func(1), E, abs_tol=1e-9)
    assert power_10(3) == 1000


def test_powers():
    assert square(5) == 25
    assert cube(3) == 27
    assert power_xy(2, 8) == 256


def test_scientific_errors():
    with pytest.raises(ValueError, match="Invalid logarithm input"):
        log10_func(0)
    with pytest.raises(ValueError, match="Invalid natural logarithm input"):
        ln_func(-5)
    with pytest.raises(ValueError, match="Tangent undefined"):
        tan_func(90, mode="deg")
