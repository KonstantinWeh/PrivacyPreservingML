from math import isqrt
import random
import numpy as np
# helper functions


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

# ---------------


def setup(length, p):
    # setup function for ipfe
    # (G, p, g) <- GroupGen(1^l) (p passed as parameter)
    # and s = (s_1, ..., s_l) <- Z_l^p
    # return mpk = (h_i = g^si) and msk = s
    # see Simple Functional Encryption Schemes for Inner Products, page 8


    g = find_generator(p)
    s = [random.randrange(1, p - 1) for _ in range(length)]
    h = [pow(g, si, p) for si in s] 

    mpk = {"h": h, "g": g, "p": p}
    msk = s

    return mpk, msk


def encrypt(mpk, x):
    # key mpk and message x = (x1, ..., xl) e Z_p
    # chooses a random r <- Zp and computes ct0 = g^r and, for each iel, ct_i = h_i^r * g^xi . Then
    # the algorithm returns the ciphertext Ct = (ct_0; (ct_i)iel)
    # see Simple Functional Encryption Schemes for Inner Products, page 8

    r = random.randrange(1, p - 1)

    ct0 = pow(mpk["g"], r, p)

    # g = mpk["g"]
    # mpk["h"] = [h_i] = [g^(s_i)]
    # cti = (h_i)^r * g^(x_i)
    ct = [(pow(h_i, r, p) * pow(mpk["g"], x[i] % (p - 1), p)) % p for h_i, i in zip(mpk["h"], range(len(x)))]

    return (ct0, ct)


def key_der(msk, y , p):
    # sk_y = <s, y> mod (p-1)
    # just evaluating the key
    return sum((si * yi) for si, yi in zip(msk, y)) % (p - 1)


def decrypt(mpk, ct, sk_y, y):
    p, g = mpk["p"], mpk["g"]
    ct0, cts = ct

    # compute: prod_i ct_i^{y_i} / ct0^{sk_y}  = g^{<x,y>}
    num = 1
    for ci, yi in zip(cts, y):
        num = (num * pow(ci, yi, p)) % p
    denom = pow(ct0, sk_y, p)
    val = (num * inv_mod(denom, p)) % p
    # discrete log base g to recover <x,y>  (works for small message ranges)
    ip = bsgs(g, val, p)
    if ip is None:
        raise ValueError("discrete log failed (increase prime or reduce message range).")
    return ip % (p - 1)

class FunctionalEncryptionDemo:
    def __init__(self, p, x, y):
        self.p = p
        self.x = x
        self.y = y
        self.length = len(x)
        self.mpk = None
        self.msk = None
        self.Ct = None
        self.sk_y = None
        self.ip = None

    def setup(self):
        self.mpk, self.msk = setup(self.length, self.p)

    def encrypt(self):
        self.Ct = encrypt(self.mpk, self.x)

    def key_derive(self):
        self.sk_y = key_der(self.msk, self.y, self.p)

    def decrypt(self):
        self.ip = decrypt(self.mpk, self.Ct, self.sk_y, self.y)

    def run(self):
        self.setup()
        self.encrypt()
        self.key_derive()
        self.decrypt()
        print("p:", self.p, "g:", self.mpk["g"])
        print("x:", self.x)
        print("y:", self.y)
        print("<x, y> (expected):", sum(xi * yi for xi, yi in zip(self.x, self.y)))
        print("<x, y> (decrypted):", self.ip)


if __name__ == "__main__":
    # choose prime (1^lamda)
    p = 104729

    # Encrypted vector
    x = [0, 1, 2, 3]
    # Calc vector
    y = [1, 2, 1, 2]

    fe_demo = FunctionalEncryptionDemo(p, x, y)
    fe_demo.run()



    