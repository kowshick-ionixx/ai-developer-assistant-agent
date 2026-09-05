"""
Tests for is_prime.py.

Covers:
- negative numbers
- 0
- 1
- 2
- 3
- composite numbers
- larger prime numbers
- invalid types (TypeError)
"""

import pytest

from is_prime import is_prime


@pytest.mark.parametrize("n", [-1, -5, -100, -7919])
def test_is_prime_negative_numbers(n):
    assert is_prime(n) is False


def test_is_prime_zero():
    assert is_prime(0) is False


def test_is_prime_one():
    assert is_prime(1) is False


def test_is_prime_two():
    assert is_prime(2) is True


def test_is_prime_three():
    assert is_prime(3) is True


@pytest.mark.parametrize("n", [4, 6, 8, 9, 10, 15, 25, 50, 100, 999])
def test_is_prime_composite_numbers(n):
    assert is_prime(n) is False


@pytest.mark.parametrize("n", [5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101, 7919])
def test_is_prime_larger_primes(n):
    assert is_prime(n) is True


@pytest.mark.parametrize("invalid", [2.5, "7", None, [13]])
def test_is_prime_invalid_type_raises_type_error(invalid):
    with pytest.raises(TypeError):
        is_prime(invalid)
