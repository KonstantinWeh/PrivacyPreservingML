import random
import numpy as np
from utils.math_helper import find_generator, mod_pow_numba, mod_inv_numba, bsgs_numba
from numba import njit

# Final optimized version of the IPFE scheme
# used in optimized approaches of IPFE-CNN inference

@njit
def decrypt_fast(ct0, cts, sk_y, y, p):
    num = 1
    for i in range(len(cts)):
        ci = cts[i]
        yi = y[i]
        num = (num * mod_pow_numba(ci, yi % (p - 1), p)) % p
    denom = mod_pow_numba(ct0, sk_y % (p - 1), p)
    val = (num * mod_inv_numba(denom, p)) % p
    return val

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

    def encrypt(self, x):
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        r = random.randrange(1, self.p - 1)
        ct0 = pow(int(self.g), int(r), int(self.p))

        ct = [(pow(int(h_i), int(r), int(self.p)) * pow(int(self.g), int(x_i) % (int(self.p) - 1), int(self.p))) % int(
            self.p)
              for h_i, x_i in zip(self.mpk, x)]

        return ct0, ct

    def key_derive(self, y):
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

    def decrypt(self, ct, sk_y, y, max_ip=None):
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



    