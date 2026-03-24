"""
Paillier-based Inner Product Functional Encryption (IPFE)
Based on: Agrawal, Libert, Stehlé – "Fully Secure Functional Encryption for Inner
Products, from Standard Assumptions" (Crypto 2016), Section 5.1.

Scheme overview (integer inner products over Z):
------------------------------------------------
Setup:
  - Generate safe RSA modulus  N = p*q  (p = 2p'+1, q = 2q'+1)
  - Pick g' <-R Z*_{N^2}, compute  g = g'^{2N} mod N^2
    (g generates the subgroup of (2N)-th residues in Z*_{N^2})
  - Sample master secret key  s = (s_1,...,s_l) <- Gaussian(σ)
  - Compute  h_i = g^{s_i} mod N^2   for each i
  - mpk = (N, g, {h_i}, Y_bound),  msk = {s_i}

KeyGen(msk, y):
  - sk_y = <s, y> = sum(s_i * y_i)   over Z  (not reduced!)

Encrypt(mpk, x):
  - r  <- {0,...,floor(N/4)}
  - C_0 = g^r mod N^2
  - C_i = (1 + N)^{x_i} * h_i^r mod N^2   for each i
  - ciphertext = (C_0, C_1,...,C_l)

Decrypt(mpk, sk_y, ct, y):
  - C_y = (prod_i  C_i^{y_i}) * C_0^{-sk_y}  mod N^2
  - output  L(C_y)  where  L(u) = (u-1)/N mod N   (discrete log in subgroup 1+N*Z_N)

Security: DCR assumption (Decision Composite Residuosity).
The scheme is FULLY (adaptively) secure – unlike the DDH construction of Abdalla et al.
that was only selectively secure.

Constraints (for correctness):
  - ‖x‖_∞ ≤ X_bound,  ‖y‖_∞ ≤ Y_bound
  - X_bound * Y_bound * l  <  N   (so <x,y> is recovered exactly mod N = over Z)
"""

import random
import math
from phe import paillier


# ---------------------------------------------------------------------------
# Low-level Paillier helpers (thin wrappers so we stay close to the paper)
# ---------------------------------------------------------------------------

def _L(u: int, N: int) -> int:
    """L function: L(u) = (u - 1) // N   (works exactly when u ≡ 1 mod N)."""
    return (u - 1) // N


def _paillier_encrypt_raw(g: int, N: int, N2: int, x_i: int, r: int) -> int:
    """
    Encrypt a single integer x_i:
        C_i = (1 + N)^{x_i} * h_i^r mod N^2
    Here we defer the h_i^r factor to the caller so this just
    computes the (1+N)^{x_i} part.
    """
    return pow(1 + N, x_i % N, N2)


def _paillier_decrypt_raw(C: int, sk: int, N: int, N2: int) -> int:
    """
    Recover plaintext from a raw Paillier ciphertext C = (1+N)^m * r^N:
        L(C^sk mod N^2) / L(g^sk mod N^2)
    For our scheme, g generates the (2N)-th residues, so we use the
    standard Paillier decryption shortcut with the factorization.
    """
    # L(C^sk mod N^2) * mu mod N  where mu = L(g^sk)^{-1} mod N
    # Since we built g = g'^{2N}, we have g^r = (g')^{2Nr} which is a
    # perfect N-th power, so the standard Paillier L trick works.
    num = _L(pow(C, sk, N2), N) % N
    return num


# ---------------------------------------------------------------------------
# IPFE-Paillier class
# ---------------------------------------------------------------------------

class IPFEPaillier:
    """
    Paillier-based IPFE following Section 5.1 of Agrawal-Libert-Stehlé 2016.

    Parameters
    ----------
    n_length : int
        Bit-length of the RSA modulus N = p*q.  Use ≥ 2048 for real security;
        smaller values (e.g. 512) are fine for fast testing.
    gaussian_sigma : float | None
        Std-dev for the discrete Gaussian secret key distribution.
        If None, we fall back to a uniform sample in [1, N^(1/4)] which is
        sufficient for correctness tests (the paper requires σ > sqrt(λ)*N^(5/2)).
    """

    def __init__(self, n_length: int = 512, gaussian_sigma=None):
        self.n_length = n_length
        self.gaussian_sigma = gaussian_sigma

        # Populated by setup()
        self.N = None
        self.N2 = None
        self.g = None          # generator of (2N)-th residues
        self.mpk = None        # list of h_i = g^{s_i} mod N^2
        self.msk = None        # list of s_i  (integer Gaussian samples)
        self.length = None
        self.Y_bound = None

        # phe keypair (used internally for N, p, q)
        self._pk = None
        self._sk = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, l: int, X_bound: int = 255, Y_bound: int = 10000):
        """
        Run the setup algorithm.

        Parameters
        ----------
        l        : vector dimension
        X_bound  : max absolute value of plaintext entries  (‖x‖_∞ ≤ X_bound)
        Y_bound  : max absolute value of key entries        (‖y‖_∞ ≤ Y_bound)
        """
        # Correctness requires  l * X_bound * Y_bound  <  N
        # We check this after key generation.

        self.length = l
        self.Y_bound = Y_bound

        # --- Generate safe RSA modulus via phe ----------------------------
        self._pk, self._sk = paillier.generate_paillier_keypair(
            n_length=self.n_length
        )
        N = self._pk.n
        N2 = N * N
        self.N = N
        self.N2 = N2

        # --- Build generator g of the (2N)-th residue subgroup -----------
        # Pick g' <-R Z*_{N^2}, then g = g'^{2N} mod N^2.
        # The subgroup of (2N)-th residues has order  phi(N^2)/(2N)
        # and contains all elements of the form (1+N)^a * something^N.
        g_prime = random.randint(2, N2 - 1)
        while math.gcd(g_prime, N2) != 1:
            g_prime = random.randint(2, N2 - 1)
        g = pow(g_prime, 2 * N, N2)
        self.g = g

        # --- Sample master secret key s -----------------------------------
        if self.gaussian_sigma is not None:
            # Discrete Gaussian approximation via Box-Muller
            s = [
                int(random.gauss(0, self.gaussian_sigma))
                for _ in range(l)
            ]
        else:
            # Simple uniform in [-N^{1/4}, N^{1/4}] for fast testing
            bound = max(1, int(N ** 0.25))
            s = [random.randint(-bound, bound) for _ in range(l)]

        self.msk = s

        # --- Compute public key  h_i = g^{s_i} mod N^2 -------------------
        h = [pow(g, s_i, N2) for s_i in s]
        self.mpk = h

        # --- Sanity check -------------------------------------------------
        max_ip = l * X_bound * Y_bound
        if max_ip >= N:
            raise ValueError(
                f"Correctness violated: l*X_bound*Y_bound = {max_ip} >= N = {N}.\n"
                f"Increase n_length or reduce X_bound / Y_bound."
            )

        print(f"=== IPFEPaillier Setup ===")
        print(f"  n_length : {self.n_length} bits")
        print(f"  N        : {N.bit_length()} bits")
        print(f"  l        : {l}")
        print(f"  X_bound  : {X_bound}")
        print(f"  Y_bound  : {Y_bound}")
        print(f"  max |<x,y>| ≤ {max_ip}  <  N ✓")

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def key_derive(self, y: list) -> int:
        """
        KeyGen(msk, y) = <s, y>  over  Z  (no modular reduction!).

        The result is a single integer used as the Paillier decryption exponent.
        """
        if len(y) != self.length:
            raise ValueError(f"y has length {len(y)}, expected {self.length}")
        return sum(si * yi for si, yi in zip(self.msk, y))

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------

    def encrypt(self, x: list) -> tuple:
        """
        Encrypt(mpk, x) -> (C_0, [C_1,...,C_l])

        C_0   = g^r mod N^2
        C_i   = (1+N)^{x_i} * h_i^r mod N^2
        """
        if len(x) != self.length:
            raise ValueError(f"x has length {len(x)}, expected {self.length}")

        N, N2 = self.N, self.N2

        r = random.randint(0, N // 4)
        C0 = pow(self.g, r, N2)

        ct = []
        for x_i, h_i in zip(x, self.mpk):
            # (1+N)^{x_i} mod N^2  — supports negative x_i via modular arithmetic
            msg_part = (1 + (int(x_i) * N)) % N2
            key_part = pow(h_i, r, N2)
            C_i = (msg_part * key_part) % N2
            ct.append(C_i)

        return C0, ct

    # ------------------------------------------------------------------
    # Decryption
    # ------------------------------------------------------------------

    def decrypt(self, ciphertext: tuple, sk_y: int, y: list) -> int:
        """
        Decrypt(mpk, sk_y, ct, y) -> <x, y>  over Z

        Algorithm:
            C_y = (prod_i  C_i^{y_i}) * C_0^{-sk_y}  mod N^2
            output  L(C_y) = (C_y - 1) / N  mod N

        The result is in [-(N//2), N//2], interpreted as a signed integer.
        """
        N, N2 = self.N, self.N2
        C0, ct = ciphertext

        if len(ct) != self.length:
            raise ValueError("Ciphertext length mismatch.")
        if len(y) != self.length:
            raise ValueError("y length mismatch.")

        # prod_i  C_i^{y_i}  mod N^2
        numerator = 1
        for C_i, y_i in zip(ct, y):
            if y_i != 0:
                numerator = (numerator * pow(C_i, int(y_i), N2)) % N2

        # C_0^{sk_y}  mod N^2
        denom = pow(C0, int(sk_y), N2)
        denom_inv = pow(denom, -1, N2)   # modular inverse

        C_y = (numerator * denom_inv) % N2

        # L(C_y) = (C_y - 1) / N  — exact integer division
        val = _L(C_y, N) % N

        # Interpret as signed integer in [-(N//2), N//2]
        if val > N // 2:
            val -= N

        return val

    # ------------------------------------------------------------------
    # Convenience run() method  (mirrors the DDH-IPFE interface)
    # ------------------------------------------------------------------

    def run(self, x: list, y: list, bias: float = 0.0, scale: int = 1):
        """
        Full encrypt → key_derive → decrypt cycle.

        scale : multiply y before encryption to preserve fractional values,
                then divide the result.  E.g. scale=10000 keeps 4 decimal places.
        """
        # Scale y to integers
        y_scaled = [int(round(yi * scale)) for yi in y]

        ct      = self.encrypt(x)
        sk_y    = self.key_derive(y_scaled)
        ip_raw  = self.decrypt(ct, sk_y, y_scaled)
        ip      = ip_raw / scale

        expected = sum(xi * yi for xi, yi in zip(x, y))

        print("\n=== IPFE-Paillier Run ===")
        print(f"  x        : {x}")
        print(f"  y        : {y}")
        print(f"  scale    : {scale}")
        print(f"  y_scaled : {y_scaled}")
        print(f"  <x,y> expected  : {expected + bias:.6f}")
        print(f"  <x,y> decrypted : {ip + bias:.6f}")
        print(f"  error           : {abs(expected - ip):.2e}")
        return ip + bias


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Same inputs as the DDH-IPFE demo ---------------------------------
    x_input = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    y_input = [-0.0300, -0.0261, -0.0194, 0.0786, 0.3495,
               -0.0135, -0.3420, -0.1810, -0.2126]
    bias    = -0.016008036211133003
    scale   = 10000

    # Bounds for correctness check
    X_bound = max(abs(v) for v in x_input)          # 254
    Y_bound = max(abs(int(round(v * scale))) for v in y_input)  # 3495
    l       = len(x_input)                          # 9

    print(f"l={l}, X_bound={X_bound}, Y_bound={Y_bound}")
    print(f"max |<x,y>| ≤ {l * X_bound * Y_bound}")

    # n_length=512 gives N ~ 2^512; well above l*X*Y ~ 10M
    ipfe = IPFEPaillier(n_length=512)
    ipfe.setup(l, X_bound=X_bound, Y_bound=Y_bound)
    ipfe.run(x_input, y_input, bias=bias, scale=scale)