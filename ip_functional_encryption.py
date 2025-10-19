import random

import numpy as np

from math_helper import inv_mod, factor, bsgs, find_generator
import matplotlib.pyplot as plt

class IPFE:
    def __init__(self, p):
        self.p = p
        self.g = None
        self.length = None
        self.mpk = None
        self.msk = None

    # ✅ Checked
    def setup(self, l):
        # setup function for ipfe
        # (G, p, g) <- GroupGen(1^l) (p passed as parameter)
        # and s = (s_1, ..., s_l) <- Z_l^p
        # return mpk = (h_i = g^si) and msk = s
        # see Simple Functional Encryption Schemes for Inner Products, page 8
        self.length = l
        self.g = find_generator(self.p)
        s = [random.randrange(1, self.p - 1) for _ in range(self.length)]
        h = [pow(self.g, s_i, self.p) for s_i in s]

        self.mpk = h
        self.msk = s

    # ✅ Checked
    def encrypt(self, x):
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        r = random.randrange(1, self.p - 1)

        ct0 = pow(self.g, r, self.p)

        # mpk = [h_0 ... h_n] = [h_i] = [g^(s_i)]
        # ct_i = (h_i)^r * g^(x_i) mod p
        ct = [(pow(h_i, r, self.p) * pow(self.g, x_i % (self.p-1), self.p)) % self.p for h_i, x_i in zip(self.mpk, x)]

        return ct0, ct

    # ✅ Checked
    def key_derive(self, y):
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

    def decrypt(self, ct, sk_y, y):
        ct0, cts = ct

        # prod_i ct_i^{y_i}
        num = 1
        for ci, yi in zip(cts, y):
            num = (num * pow(ci, yi % (self.p - 1), self.p)) % self.p

        # ct0^{sk_y}
        denom = pow(ct0, sk_y % (self.p - 1), self.p)

        # compute: prod_i ct_i^{y_i} / ct0^{sk_y}  = g^{<x,y>}
        val = (num * inv_mod(denom, self.p)) % self.p

        # discrete log base g to recover <x,y>  (works for small message ranges)
        ip = bsgs(self.g, val, self.p)
        if ip is None:
            raise ValueError("discrete log failed (increase prime or reduce message range).")

        return ip

    def run(self, l, x, y):
        self.setup(l)
        # ct_0, ct
        ct = self.encrypt(x)
        plt.imshow(np.asarray(ct[1]).reshape(28, 28), cmap='gray')
        plt.title("Encrypted image")
        plt.show()
        plt.close()

        sk_y = self.key_derive(y)
        ip = self.decrypt(ct, sk_y, y)
        print("p:", self.p, "g:", self.g)
        print("x:", x)
        print("y:", y)
        print("<x, y> (expected):", sum((xi * yi) for xi, yi in zip(x, y)) % (self.p - 1))
        print("<x, y> (decrypted):", ip)


if __name__ == "__main__":
    # choose prime (1^lamda)
    p_input = 104729
    # p_input = 67
    # Encrypted vector
    x_input = [1000000, 100, 200, 300, 5000]
    # Calc vector
    y_input = [1, 2, 1, 2, 200]

    ipfe_demo = IPFE(p_input)
    ipfe_demo.run(len(x_input), x_input, y_input)



    