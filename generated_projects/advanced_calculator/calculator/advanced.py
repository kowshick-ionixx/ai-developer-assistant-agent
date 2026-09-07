"""Advanced calculator operations (Combinations, Permutations, GCD, LCM, etc.)."""
import math

def combinations(n: int, r: int) -> int:
    if not (isinstance(n, int) and isinstance(r, int)):
        try:
            n = int(n)
            r = int(r)
        except Exception:
            raise ValueError("n and r must be integers")
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    if r > n:
        raise ValueError("r cannot be greater than n in combinations")
    return math.comb(n, r)

def permutations(n: int, r: int) -> int:
    if not (isinstance(n, int) and isinstance(r, int)):
        try:
            n = int(n)
            r = int(r)
        except Exception:
            raise ValueError("n and r must be integers")
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    if r > n:
        raise ValueError("r cannot be greater than n in permutations")
    return math.perm(n, r)

def gcd_func(a: int, b: int) -> int:
    try:
        ia, ib = int(a), int(b)
    except Exception:
        raise ValueError("GCD arguments must be integers")
    return math.gcd(ia, ib)

def lcm_func(a: int, b: int) -> int:
    try:
        ia, ib = int(a), int(b)
    except Exception:
        raise ValueError("LCM arguments must be integers")
    if ia == 0 and ib == 0:
        return 0
    return abs(ia * ib) // math.gcd(ia, ib)

def floor_func(x: float) -> int:
    return math.floor(x)

def ceiling_func(x: float) -> int:
    return math.ceil(x)

def scientific_notation(x: float, decimals: int = 4) -> str:
    if not math.isfinite(x):
        return str(x)
    return f"{x:.{decimals}e}"

# Aliases matching the names app.py imports
nCr = combinations
nPr = permutations
gcd_val = gcd_func
lcm_val = lcm_func
floor_val = floor_func
ceil_val = ceiling_func
sci_notation = scientific_notation
