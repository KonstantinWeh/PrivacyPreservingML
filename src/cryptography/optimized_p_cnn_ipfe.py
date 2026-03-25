import random
import math
import time
import numpy as np
import concurrent.futures
from phe import paillier
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _encrypt_worker(args):
    """Worker for encrypt_batch -- picklable top-level function."""
    x, g, h, N, N2 = args
    r = random.randint(0, N // 4)
    C0 = pow(g, r, N2)
    ct = []
    for xi, hi in zip(x, h):
        # (1+N)^{x_i} = 1 + x_i*N  (mod N^2)  — linearisation valid for |x_i| << N
        msg_part = (1 + int(xi) * N) % N2
        key_part = pow(hi, r, N2)
        ct.append((msg_part * key_part) % N2)
    return C0, ct


def _decrypt_worker(args):
    """Worker for decrypt_patches_batch -- picklable top-level function."""
    C0, ct, sk_y, y, N, N2 = args

    # C_0^{-sk_y} mod N^2
    denom_inv = pow(pow(C0, int(sk_y), N2), -1, N2)

    # prod_i  C_i^{y_i}  mod N^2
    num = 1
    for Ci, yi in zip(ct, y):
        if yi != 0:
            num = (num * pow(Ci, int(yi), N2)) % N2

    # C_y = numerator * denom_inv  mod N^2
    Cy = (num * denom_inv) % N2

    # L(C_y) = (C_y - 1) / N mod N  — exact integer division (no discrete log!)
    val = ((Cy - 1) // N) % N

    # Signed adjustment: map to [-(N//2), N//2]
    if val > N // 2:
        val -= N

    return int(val)

# ---------------------------------------------------------------------------
# IPFEPaillier
# ---------------------------------------------------------------------------

class IPFEPaillier:
    """
    Paillier-based IPFE -- Section 5.1 of Agrawal-Libert-Stehle (Crypto 2016).

    Drop-in replacement for the DDH-IPFE with the same public interface,
    plus batch helpers that mirror decrypt_patches_batch from the Numba IPFE.

    Key differences vs DDH-IPFE
    ----------------------------
    * No discrete log in decryption  ->  no BSGS, no inner-product range limit
      (range is bounded only by N, not by sqrt(p)).
    * Fully adaptive security (DCR) vs selective security (DDH).
    * Encryption uses the (1+N)^x = 1+xN (mod N^2) linearisation -- ~60x faster
      than a general modular exponentiation for small |x|.
    * sk_y = <s, y> over Z with NO modular reduction  (critical for correctness).
    * Batch ops use ProcessPoolExecutor instead of Numba @njit because Python's
      built-in pow() for big integers already uses GMP and cannot be compiled
      by Numba (which is limited to fixed-width numeric types).

    Parameters
    ----------
    n_length      : RSA modulus bit-length. >= 2048 for real security; 512 for tests.
    gaussian_sigma: Std-dev of the discrete Gaussian master secret.
                    None -> uniform in [-N^{1/4}, N^{1/4}] (correctness-only).
    max_workers   : Pool size for batch operations. None = os.cpu_count().
    """

    def __init__(self, n_length: int = 512, gaussian_sigma=None, max_workers=4):
        self.n_length        = n_length
        self.gaussian_sigma  = gaussian_sigma
        self.max_workers     = max_workers

        # Populated by setup()
        self.N       = None
        self.N2      = None
        self.g       = None      # generator of the (2N)-th residue subgroup
        self.mpk     = None      # [h_i = g^{s_i} mod N^2]
        self.msk     = None      # [s_i] integer Gaussian (or uniform) samples
        self.length  = None
        self.Y_bound = None

        self._pk = None          # phe PaillierPublicKey  (for N)
        self._sk = None          # phe PaillierPrivateKey (for p, q)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, l: int, X_bound: int = 255, Y_bound: int = 10000):
        """
        Setup(1^lambda, 1^l, X, Y).

        Parameters
        ----------
        l       : vector dimension
        X_bound : ||x||_inf upper bound (plaintext entries)
        Y_bound : ||y||_inf upper bound (key entries, after scaling)
        """
        self.length  = l
        self.Y_bound = Y_bound

        # Generate RSA modulus N = p*q (phe handles safe-prime generation)
        self._pk, self._sk = paillier.generate_paillier_keypair(n_length=self.n_length)
        N  = self._pk.n
        N2 = N * N
        self.N  = N
        self.N2 = N2

        # g = g'^{2N} mod N^2  -- generator of (2N)-th residue subgroup
        g_prime = random.randint(2, N2 - 1)
        while math.gcd(g_prime, N2) != 1:
            g_prime = random.randint(2, N2 - 1)
        self.g = pow(g_prime, 2 * N, N2)

        # Master secret key: integer Gaussian (or uniform for fast testing)
        if self.gaussian_sigma is not None:
            s = [int(random.gauss(0, self.gaussian_sigma)) for _ in range(l)]
        else:
            bound = max(1, int(N ** 0.25))
            s = [random.randint(-bound, bound) for _ in range(l)]
        self.msk = s

        # Public key: h_i = g^{s_i} mod N^2
        self.mpk = [pow(self.g, s_i, self.N2) for s_i in s]

        # Correctness guard: l * X * Y must be < N
        max_ip = l * X_bound * Y_bound
        if max_ip >= N:
            raise ValueError(
                f"Correctness violated: l*X_bound*Y_bound = {max_ip} >= N.\n"
                f"Increase n_length or reduce X_bound / Y_bound."
            )

        print(f"=== IPFEPaillier Setup ===")
        print(f"  n_length  : {self.n_length} bits")
        print(f"  N         : {N.bit_length()} bits")
        print(f"  l         : {l}")
        print(f"  X_bound   : {X_bound}")
        print(f"  Y_bound   : {Y_bound}")
        print(f"  max|<x,y>|: {max_ip}  <  N ok")

    # ------------------------------------------------------------------
    # Single-sample operations
    # ------------------------------------------------------------------

    def key_derive(self, y: list) -> int:
        """
        KeyGen(msk, y) = <s, y>  over Z -- no modular reduction.

        The scalar result is used as the decryption exponent.
        Unlike DDH-IPFE there is no (p-1) modulus here.
        """
        if len(y) != self.length:
            raise ValueError(f"y has length {len(y)}, expected {self.length}")
        return sum(si * yi for si, yi in zip(self.msk, y))

    def encrypt(self, x: list) -> tuple:
        """
        Encrypt(mpk, x)  ->  (C_0, [C_1,...,C_l])

            C_0 = g^r  mod N^2
            C_i = (1 + x_i*N) * h_i^r  mod N^2   [linearised]
        """
        if len(x) != self.length:
            raise ValueError(f"x has length {len(x)}, expected {self.length}")
        return _encrypt_worker((x, self.g, self.mpk, self.N, self.N2))

    def decrypt(self, ciphertext: tuple, sk_y: int, y: list) -> int:
        """
        Decrypt(mpk, sk_y, ct, y)  ->  <x, y>  over Z

            C_y = (prod_i C_i^{y_i}) * C_0^{-sk_y}  mod N^2
            output L(C_y) = (C_y - 1) / N  mod N

        No discrete log required (unlike DDH-IPFE / BSGS).
        Returns a signed integer in [-(N//2), N//2].
        """
        if len(y) != self.length:
            raise ValueError(f"y has length {len(y)}, expected {self.length}")
        C0, ct = ciphertext
        return _decrypt_worker((C0, ct, sk_y, y, self.N, self.N2))

    # ------------------------------------------------------------------
    # Batch operations  (mirrors decrypt_patches_batch from Numba-IPFE)
    # ------------------------------------------------------------------

    def encrypt_batch(self, x_list: list) -> list:
        """
        Encrypt a list of plaintext vectors in parallel.

        Parameters
        ----------
        x_list : list of l-dimensional integer lists

        Returns
        -------
        list of (C0, ct) ciphertext tuples
        """
        args = [(x, self.g, self.mpk, self.N, self.N2) for x in x_list]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(_encrypt_worker, args))

    def decrypt_patches_batch(self,
                              ct_list: list,
                              sk_y: int,
                              y: list,
                              scale: int = 1) -> np.ndarray:
        """
        Batch-decrypt a list of ciphertexts.

        Mirrors decrypt_patches_batch from the Numba DDH-IPFE.
        Uses ProcessPoolExecutor instead of @njit because Python's big-integer
        pow() cannot be compiled by Numba (it requires GMP-level arithmetic
        on 512-1024 bit numbers, beyond Numba's fixed-width type system).

        Parameters
        ----------
        ct_list : list of (C0, ct) tuples  (e.g. from encrypt_batch or encrypt)
        sk_y    : scalar functional secret key  (from key_derive)
        y       : scaled integer key vector
        scale   : divide results by this to recover float inner products

        Returns
        -------
        numpy array of (signed) inner products, divided by scale if scale != 1
        """
        args = [(C0, ct, sk_y, y, self.N, self.N2) for (C0, ct) in ct_list]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(_decrypt_worker, args))
        arr = np.array(results, dtype=np.int64)
        return (arr.astype(np.float32) / scale) if scale != 1 else arr.astype(np.float32)

    def decrypt_patches_kernels_batch(self,
                                      ct_list: list,
                                      sk_y_list: list,
                                      y_list: list,
                                      scale: int = 1) -> np.ndarray:
        """
        Batch-decrypt a list of ciphertexts for multiple kernels.

        Parameters
        ----------
        ct_list   : list of (C0, ct) tuples
        sk_y_list : list of scalar functional secret keys
        y_list    : list of scaled integer key vectors
        scale     : divide results by this to recover float inner products

        Returns
        -------
        numpy array of shape (num_patches, num_kernels) with (signed) inner products
        """
        num_patches = len(ct_list)
        num_kernels = len(sk_y_list)
        args = []
        for i in range(num_patches):
            C0, ct = ct_list[i]
            for k in range(num_kernels):
                args.append((C0, ct, sk_y_list[k], y_list[k], self.N, self.N2))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(_decrypt_worker, args))

        arr = np.array(results, dtype=np.int64).reshape(num_patches, num_kernels)
        return (arr.astype(np.float32) / scale) if scale != 1 else arr.astype(np.float32)

    def decrypt_patches_kernels_batch_optimized(self,
                                      ct_list: list,
                                      sk_y_list: list,
                                      y_list: list,
                                      scale: int = 1) -> np.ndarray:
        """
        Optimized batch-decrypt: group patches per kernel to reduce task overhead.

        Instead of submitting num_patches × num_kernels tiny tasks,
        submit num_kernels tasks where each processes ALL patches for one kernel.

        Parameters
        ----------
        ct_list   : list of (C0, ct) tuples
        sk_y_list : list of scalar functional secret keys
        y_list    : list of scaled integer key vectors
        scale     : divide results by this to recover float inner products

        Returns
        -------
        numpy array of shape (num_patches, num_kernels) with (signed) inner products
        """
        num_patches = len(ct_list)
        num_kernels = len(sk_y_list)

        def decrypt_kernel_all_patches(k_idx):
            """Decrypt all patches for a single kernel (runs in thread)."""
            sk_y = sk_y_list[k_idx]
            y = y_list[k_idx]
            results = np.zeros(num_patches, dtype=np.int64)
            for p in range(num_patches):
                C0, ct = ct_list[p]
                results[p] = _decrypt_worker((C0, ct, sk_y, y, self.N, self.N2))
            return results

        # Submit only num_kernels tasks (one per kernel), each handling all patches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(decrypt_kernel_all_patches, range(num_kernels)))

        # Transpose: results[k] is a 1D array of patch values
        arr = np.array(results, dtype=np.int64)  # shape: (num_kernels, num_patches)
        arr = arr.T  # transpose to (num_patches, num_kernels)
        return (arr.astype(np.float32) / scale) if scale != 1 else arr.astype(np.float32)

    def key_derive_batch(self, y_matrix) -> np.ndarray:
        """
        Derive secret keys for multiple y vectors at once.

        Parameters
        ----------
        y_matrix : (K, l) array-like of integer key vectors

        Returns
        -------
        1-D array of K scalar secret keys  sk_{y_k} = <s, y_k>
        """
        Y = np.array(y_matrix, dtype=object)   # object dtype handles big-int safely
        s = np.array(self.msk, dtype=object)
        return np.array([int(np.dot(s, Y[k])) for k in range(len(Y))])

    # ------------------------------------------------------------------
    # Convenience run() -- full cycle, mirrors old IPFE.run()
    # ------------------------------------------------------------------

    def run(self, x: list, y: list, bias: float = 0.0, scale: int = 1,
            image: bool = False) -> float:
        """
        Full  encrypt -> key_derive -> decrypt  cycle.

        Parameters
        ----------
        x     : plaintext integer vector
        y     : weight vector (floats; scaled to ints internally)
        bias  : added to the result for display (e.g. layer bias term)
        scale : multiply y entries before key derivation, divide result after.
                Use e.g. scale=10000 to keep 4 decimal places of float weights.
        image : if True, visualise ciphertext component fingerprint as a
                greyscale image (works best when l == 784 for 28x28 patches).
        """
        y_scaled = [int(round(yi * scale)) for yi in y]

        t0 = time.perf_counter()
        ct = self.encrypt(x)
        t_enc = time.perf_counter() - t0

        if image:
            _, ct_components = ct
            vis = np.array([int(c) % (2**16) for c in ct_components], dtype=np.float32)
            vis = (vis / vis.max() * 255).astype(np.uint8)
            side = int(math.isqrt(len(vis)))
            if side * side == len(vis):
                plt.imshow(vis.reshape(side, side), cmap='gray')
            else:
                plt.plot(vis)
            plt.title("Ciphertext fingerprint")
            plt.show()
            plt.close()

        t0 = time.perf_counter()
        sk_y = self.key_derive(y_scaled)
        t_kg = time.perf_counter() - t0

        t0 = time.perf_counter()
        ip_raw = self.decrypt(ct, sk_y, y_scaled)
        t_dec = time.perf_counter() - t0

        ip       = ip_raw / scale
        expected = sum(xi * yi for xi, yi in zip(x, y))

        print(f"\n=== IPFE-Paillier Run ===")
        print(f"  x               : {x}")
        print(f"  y               : {y}")
        print(f"  scale           : {scale}")
        print(f"  y_scaled        : {y_scaled}")
        print(f"  <x,y> expected  : {expected + bias:.6f}")
        print(f"  <x,y> decrypted : {ip + bias:.6f}")
        print(f"  error           : {abs(expected - ip):.2e}")
        print(f"  timing  enc={t_enc*1000:.1f}ms  keygen={t_kg*1000:.2f}ms  dec={t_dec*1000:.1f}ms")

        return ip + bias


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Same inputs as the DDH-IPFE demo ──────────────────────────────────
    x_input = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    y_input = [-0.0300, -0.0261, -0.0194, 0.0786, 0.3495,
               -0.0135, -0.3420, -0.1810, -0.2126]
    bias    = -0.016008036211133003
    scale   = 10000

    X_bound = max(abs(v) for v in x_input)
    Y_bound = max(abs(int(round(v * scale))) for v in y_input)
    l       = len(x_input)

    print(f"l={l}, X_bound={X_bound}, Y_bound={Y_bound}")
    print(f"max |<x,y>| <= {l * X_bound * Y_bound}")

    ipfe = IPFEPaillier(n_length=512)
    ipfe.setup(l, X_bound=X_bound, Y_bound=Y_bound)

    # ── Single sample ──────────────────────────────────────────────────────
    ipfe.run(x_input, y_input, bias=bias, scale=scale)

    # ── Batch decrypt (mirrors decrypt_patches_batch from Numba-IPFE) ──────
    print("\n=== Batch decrypt demo ===")
    y_scaled = [int(round(yi * scale)) for yi in y_input]
    sk_y = ipfe.key_derive(y_scaled)

    N_PATCHES = 64
    ct_list   = [ipfe.encrypt(x_input) for _ in range(N_PATCHES)]

    # results = ipfe.decrypt_patches_batch(ct_list, sk_y, y_scaled, scale=scale)

    # expected_ip = sum(xi * yi for xi, yi in zip(x_input, y_input))
    # print(f"  results  : {results}")
    # print(f"  expected : {expected_ip:.6f}")

    # ── Key derive batch ───────────────────────────────────────────────────
    print("\n=== key_derive_batch demo ===")
    y_matrix = [y_scaled, [yi * 2 for yi in y_scaled], [0] * l]
    sk_batch = ipfe.key_derive_batch(y_matrix)
    print(f"  sk_y (single) : {sk_y}")
    print(f"  sk_batch[0]   : {sk_batch[0]}  (should match)")
    print(f"  match         : {int(sk_batch[0]) == sk_y}")

    # ── Batch decrypt for multiple kernels ──────────────────────────────────
    print("\n=== decrypt_patches_kernels_batch demo ===")
    sk_y_list = sk_batch
    y_list = y_matrix
    results = ipfe.decrypt_patches_kernels_batch(ct_list, sk_y_list, y_list, scale=scale)
    print(f"  results shape : {results.shape}")
    expected_ip0 = sum(xi * yi for xi, yi in zip(x_input, y_input))
    expected_ip1 = sum(xi * yi * 2 for xi, yi in zip(x_input, y_input))
    expected_ip2 = 0
    print(f"  expected kernel 0: {expected_ip0:.6f}")
    print(f"  expected kernel 1: {expected_ip1:.6f}")
    print(f"  expected kernel 2: {expected_ip2:.6f}")
    print(f"  sample results[0,:]: {results[0,:]}")

    # ── Optimized batch decrypt for multiple kernels ─────────────────────────
    print("\n=== decrypt_patches_kernels_batch_optimized demo ===")
    results_opt = ipfe.decrypt_patches_kernels_batch_optimized(ct_list, sk_y_list, y_list, scale=scale)
    print(f"  results shape : {results_opt.shape}")
    assert np.array_equal(results, results_opt), "Optimized results differ!"
