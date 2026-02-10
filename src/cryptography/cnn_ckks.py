import tenseal as ts
import torch

class CKKSFHE:
    """
    Simple CKKS-based 'inner-product' helper, analogous in spirit to IPFE.IPFE:
    - setup(l): choose CKKS params & remember vector length l
    - encrypt(x): encrypt a length-l vector x
    - inner_product(ct_x, y): homomorphic <x, y> (approximate) + decrypt
    """
    def __init__(self):
        self.length = None
        self.ctx = None

    def setup(self, l, poly_modulus_degree=8192, coeff_mod_bit_sizes=None, scale_bits=30):
        """
        Initialize CKKS context and fix vector length.
        This is conceptually your 'scheme setup'; it does not depend on y.
        """
        if coeff_mod_bit_sizes is None:
            # shallow chain: enough for a few mults + rotations
            coeff_mod_bit_sizes = [50, 30, 50]

        self.length = l

        # Create TenSEAL CKKS context
        self.ctx = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        self.ctx.global_scale = 2 ** scale_bits
        self.ctx.generate_galois_keys()

    def encrypt(self, x):
        """
        Encrypt a length-l vector x (list, tensor, or iterable of floats).
        Returns a ts.ckks_vector ciphertext.
        """
        if self.length is None:
            raise ValueError("Call setup(l) before encrypt.")
        # Normalize input to Python list of floats
        if isinstance(x, torch.Tensor):
            x_list = x.detach().cpu().view(-1).tolist()
        else:
            x_list = list(x)
        if len(x_list) != self.length:
            raise ValueError("x length does not match setup length.")

        return ts.ckks_vector(self.ctx, x_list)

    def inner_product(self, ct_x, y):
        """
        ct_x: CKKSVector ciphertext of x
        y: length-l vector
        """
        if isinstance(y, torch.Tensor):
            y_list = y.detach().cpu().view(-1).tolist()
        else:
            y_list = list(y)
        if len(y_list) != self.length:
            raise ValueError("y length does not match setup length.")

        # Element-wise product in encrypted form
        ct_prod = ct_x * y_list

        # For small l, decrypt and sum in the clear (still exercises encryption)
        prod_list = ct_prod.decrypt()  # Python list of floats
        return sum(prod_list)

    # Optional: small demo to mirror IPFE.run
    def run(self, x, y, bias=0.0):
        ct = self.encrypt(x)
        print("Encrypted x (CKKSVector):", ct)
        ip_approx = self.inner_product(ct, y)
        print("x:", list(x))
        print("y:", list(y))
        print("<x,y> (expected):",
              sum(float(xi) * float(yi) for xi, yi in zip(x, y)) + bias)
        print("<x,y> (decrypted ~):", ip_approx + bias)


if __name__ == "__main__":
    x_input = torch.tensor([0., 133., 254., 9., 205., 248., 126., 254., 182.])
    y_input = torch.tensor([-0.0300, -0.0261, -0.0194,
                            0.0786, 0.3495, -0.0135,
                            -0.3420, -0.1810, -0.2126])

    fhe_demo = FHE()
    # if ipfe <= 24 or <= 30
    # 4096, 8192
    # 40, 20, 40, or 50, 30, 50
    # 20 or 25
    fhe_demo.setup(len(x_input),  poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40], scale_bits=20)
    fhe_demo.run(x_input, y_input, bias=-0.016008036211133003)

