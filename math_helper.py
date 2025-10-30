from math import isqrt

def inv_mod(a, p):
    return pow(a, p - 2, p)

def factor(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        f[n] = 1
    return list(f.keys())

def find_generator(p):
    phi = p - 1
    pf = factor(phi)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in pf):
            return g
    raise RuntimeError("no generator found")

def bsgs(g, h, p):
    """Solve g^x = h (mod p) for x in [0, p-2] (returns None if not found)."""
    m = isqrt(p) + 1
    # baby steps: g^j
    table = {}
    e = 1
    for j in range(m):
        table[e] = j
        e = (e * g) % p
    # giant step factor: g^{-m}
    gm = pow(g, m * (p - 2), p)  # g^{-m} mod p
    y = h
    for i in range(m + 1):
        if y in table:
            return i * m + table[y]
        y = (y * gm) % p
    return None