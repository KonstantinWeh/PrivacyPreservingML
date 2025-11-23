import random
import numpy as np
<<<<<<<< HEAD:src/altered_ipfe.py
from .math_helper import inv_mod, factor, bsgs, find_generator
========
from utils.math_helper import inv_mod, bsgs, find_generator, mod_pow_numba, mod_inv_numba, bsgs_numba
>>>>>>>> main:cryptography/cnn_ipfe.py
import matplotlib.pyplot as plt
import torch

class IPFE:
    def __init__(self, p):
        self.p = p
        self.g = None
        self.length = None
        self.mpk = None
        self.msk = None

    # ✅ Checked
    def setup(self, l):
        # setup function for cryptography
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
        val = (num * inv_mod(denom, self.p)) % self.p

        # discrete log base g to recover <x,y>  (works for small message ranges)
        ip = bsgs(self.g, val, self.p)
        if ip is None:
            raise ValueError("discrete log failed (increase prime or reduce message range).")

        modulus = self.p - 1
        half = modulus // 2
        if ip > half:
            ip_signed = ip - modulus
        else:
            ip_signed = ip

        return ip_signed

    def run(self, x, y, bias, scale=1, image=False):
        ct = self.encrypt(x)

        if image:
            plt.imshow(np.asarray(ct[1]).reshape(28, 28), cmap='gray')
            plt.title("Encrypted image")
            plt.show()
            plt.close()

        scaled_y_input = [int(val * scale) for val in y]

        sk_y = self.key_derive(scaled_y_input)
        ip_scaled = self.decrypt(ct, sk_y, scaled_y_input)
        ip = ip_scaled / scale

        print("p:", self.p, "g:", self.g)
        print("x:", x)
        print("y:", y)
        print("scaled y:", scaled_y_input)
        print("<x, y> (expected):", (sum((xi * yi) for xi, yi in zip(x, y)) + bias))
        print("<x, y> (decrypted):", ((ip + bias)))


if __name__ == "__main__":
    # choose prime (1^lamda)
    # p_input = 67
    #p_input = 104729
    # p_input = 1 000 000 007
    # p_input = 2300003 # appears safe but im unsure
    # p_input = 4590007  # should be 100% safe since the range of inner products is less than 4.59 million
    p_input = 1721257
    # Encrypted vector

    x_input = torch.tensor([  0., 133., 254.,   9., 205., 248., 126., 254., 182.])
    x_input = [(int(val.item()) % (p_input - 1)) for val in x_input]

    y_input = [-0.0300, -0.0261, -0.0194, 0.0786,  0.3495, -0.0135, -0.3420, -0.1810, -0.2126]
    scale = 10000

    # -67.16720803621115

    print(len(x_input))
    ipfe_demo = IPFE(p_input)
    ipfe_demo.setup(len(x_input))
    ipfe_demo.run(x_input, y_input, -0.016008036211133003, scale)



    