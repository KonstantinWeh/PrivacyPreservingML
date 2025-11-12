import random
import numpy as np
from math_helper import inv_mod, factor, bsgs, find_generator
import matplotlib.pyplot as plt
import torch

from numba import njit
import math

# --------------------------
# Helper functions
# --------------------------
@njit
def mod_pow(a, b, p):
    """Compute (a ** b) % p using binary exponentiation (Numba-friendly)."""
    result = 1
    a = a % p
    while b > 0:
        if b & 1:
            result = (result * a) % p
        a = (a * a) % p
        b >>= 1
    return result

@njit
def mod_inv(a, p):
    """Modular inverse using Fermat's little theorem (works since p is prime)."""
    return mod_pow(a, p - 2, p)

# --------------------------
# Optional Numba bsgs for small message ranges
# --------------------------
@njit
def bsgs_numba(g, h, p, max_range):
    """Discrete log solver (small message range)"""
    m = int(math.ceil(math.sqrt(max_range)))
    table = {}
    e = 1
    for j in range(m):
        table[e] = j
        e = (e * g) % p
    factor = mod_pow(g, m * (p - 2), p)  # g^-m mod p
    gamma = h
    for i in range(m):
        if gamma in table:
            return i * m + table[gamma]
        gamma = (gamma * factor) % p
    return -1

# --------------------------
# Numba-optimized modular arithmetic for decrypt
# --------------------------
@njit
def decrypt_fast(ct0, cts, sk_y, y, p):
    num = 1
    for i in range(len(cts)):
        ci = cts[i]
        yi = y[i]
        num = (num * mod_pow(ci, yi % (p - 1), p)) % p
    denom = mod_pow(ct0, sk_y % (p - 1), p)
    val = (num * mod_inv(denom, p)) % p
    return val

# -----------------------------
# Batch decrypt all patches
# -----------------------------
@njit
def decrypt_patches_batch(ct0_array, cts_array, sk_y, y_vec, g, p):
    """
    Fully Numba-decrypted patches with BSGS.
    patches_ct: list of (ct0, cts) tuples
    Returns list of signed inner products.
    """
    num_patches = len(ct0_array)
    results = np.zeros(num_patches, dtype=np.int64)

    y_vec_int = [int(y) for y in y_vec]

    for i in range(num_patches):
        ct0 = int(ct0_array[i])
        cts = [int(c) for c in cts_array[i]]

        # Step 1: modular arithmetic
        val = decrypt_fast(ct0, cts, int(sk_y), y_vec_int, p)

        # Step 2: discrete log
        ip = bsgs_numba(g, val, p, p - 1)
        if ip == -1:
            raise ValueError("Discrete log failed")

        # Step 3: signed adjustment
        modulus = p - 1
        half = modulus // 2
        if ip > half:
            ip_signed = ip - modulus
        else:
            ip_signed = ip

        results[i] = ip_signed

    return results

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

    def new_encrypt(self, x):
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        r = random.randrange(1, self.p - 1)
        ct0 = pow(int(self.g), int(r), int(self.p))

        ct = [(pow(int(h_i), int(r), int(self.p)) * pow(int(self.g), int(x_i) % (int(self.p) - 1), int(self.p))) % int(
            self.p)
              for h_i, x_i in zip(self.mpk, x)]

        return ct0, ct

    # ✅ Checked
    def key_derive(self, y):
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

    def new_decrypt(self, ct, sk_y, y, max_ip=None):
        """
        Optimized IPFE decryption.
        ct: ciphertext tuple (ct0, cts)
        sk_y: secret key derived from y
        y: scaled y vector
        Returns signed inner product.
        """

        if max_ip is None:
            max_ip = self.p - 1

        ct0, cts = ct

        # Step 1: fast modular arithmetic
        val = decrypt_fast(ct0, cts, sk_y, y, self.p)

        # Step 2: discrete log
        # You can replace with bsgs_numba if inner products are small
        ip = bsgs_numba(self.g, val, self.p, max_ip)
        if ip is None:
            raise ValueError("Discrete log failed; increase prime or reduce message range.")

        # Step 3: adjust for signed inner product
        modulus = self.p - 1
        half = modulus // 2
        if ip > half:
            ip_signed = ip - modulus
        else:
            ip_signed = ip

        return ip_signed

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
        ip_scaled = self.new_decrypt(ct, sk_y, scaled_y_input)
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



    