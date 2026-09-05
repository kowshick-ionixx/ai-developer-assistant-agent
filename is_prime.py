"""
is_prime.py
-----------
Provides a function to check whether an integer is prime.
"""


def is_prime(n: int) -> bool:
    """Check whether a number is prime.

    Returns False for numbers <= 1, True for 2 and 3, and checks divisibility
    up to sqrt(n) for larger numbers. Raises TypeError if n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
