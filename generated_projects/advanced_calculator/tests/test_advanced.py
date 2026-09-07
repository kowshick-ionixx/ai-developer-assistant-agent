import pytest
from calculator.advanced import (
    ceiling_func,
    combinations,
    floor_func,
    gcd_func,
    lcm_func,
    permutations,
    scientific_notation,
)


def test_combinations_permutations():
    assert combinations(10, 3) == 120
    assert permutations(10, 3) == 720


def test_gcd_lcm():
    assert gcd_func(48, 18) == 6
    assert lcm_func(12, 18) == 36


def test_floor_ceil():
    assert floor_func(4.7) == 4
    assert ceiling_func(4.2) == 5


def test_scientific_notation():
    assert scientific_notation(123456, decimals=2) == "1.23e+05"


def test_advanced_errors():
    with pytest.raises(ValueError):
        combinations(3, 10)
    with pytest.raises(ValueError):
        permutations(-5, 2)
