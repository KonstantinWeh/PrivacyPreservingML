import tenseal as ts
import torch

class FHE:
    def __init__(self):
        self.length = None
        self.plain_modulus = None
        self.ctx = None

    def setup(self, l, prime=1032193, poly_modulus_degree=4096):
        self.length = l
        self.plain_modulus = prime

        self.ctx = ts.context(
            scheme=ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=self.plain_modulus,
        )
        self.ctx.generate_galois_keys()

    def encrypt(self, x):
        x_list = x.detach().cpu().view(-1).tolist() if isinstance(x, torch.Tensor) else list(x)
        x_ints = [int(xi) % self.plain_modulus for xi in x_list]  # Keep as-is
        return ts.bfv_vector(self.ctx, x_ints)

    def inner_product(self, ct_x, y):
        if isinstance(y, torch.Tensor):
            y_list = y.detach().cpu().view(-1).tolist()
        else:
            y_list = list(y)

        if len(y_list) != self.length:
            raise ValueError("y length does not match setup length.")

        y_ints = [yi % self.plain_modulus for yi in y_list]

        ct_prod = ct_x * y_ints
        prod_ints = ct_prod.decrypt()
        return sum(prod_ints)

    def run(self, x, y, bias, scale=10000):
        ct = self.encrypt(x)

        scaled_y = [(int(val * scale) % self.plain_modulus) for val in y]

        print("Scaled y (mod plain_modulus):", scaled_y)
        ip_raw = self.inner_product(ct, scaled_y)
        print("Raw inner product (scaled):", ip_raw)
        ip = ip_raw / scale

        print("<x, y> (expected):", sum(xi * yi for xi, yi in zip(x, y)) + bias)
        print("<x, y> (decrypted):", ip + bias)


if __name__ == "__main__":
    plain_modulus = 1032193 # Large enough for scale=10000
    x_input = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    #y_input = [-0.03, -0.0261, -0.0194, 0.0786, 0.3495, -0.0135, -0.342, -0.181, -0.2126]
    y_input = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    fhe_demo = FHE()
    fhe_demo.setup(len(x_input), prime=plain_modulus)
    fhe_demo.run(x_input, y_input, -0.016008036211133003, scale=10000)



