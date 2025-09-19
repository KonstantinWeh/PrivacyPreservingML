# helper functions
import random


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

    mpk = h
    msk = s

    return (mpk, msk)


if __name__ == "__main__":

    # choose prime (1^lamda)
    p = 104729

    # Encrypted vector
    x_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Calc vector
    y_1 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    length = len(x_1)


    mpk, msk = setup(length, p)



    