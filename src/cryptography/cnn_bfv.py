from phe import paillier, EncryptedNumber

class BFVFHE:
    def __init__(self):
        self.length = None
        self.public_key = None
        self.private_key = None

    def setup(self, l, n_length=2048):
        self.length = l
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=n_length
        )
        print(f"\n=== Paillier Key Info ===")
        print(f"p = {self.private_key.p}")
        print(f"q = {self.private_key.q}")
        print(f"n = {self.public_key.n}")

    def encrypt(self, x):
        x_list = x.detach().cpu().view(-1).tolist() if hasattr(x, 'detach') else list(x)
        return [self.public_key.encrypt(int(xi)) for xi in x_list]

    def inner_product(self, ct_x, y):
        result = 0

        for i, (ct, y_val) in enumerate(zip(ct_x, y)):
            if isinstance(ct, EncryptedNumber):
                y_val_mod = int(y_val)
                result += ct * y_val_mod

        return self.private_key.decrypt(result)

    def run(self, x, y, bias, scale=10000):
        ct = self.encrypt(x)
        y_scaled = [int(round(yi * scale)) for yi in y]
        ip_raw = self.inner_product(ct, y_scaled)
        ip = ip_raw / scale

        print("<x,y> (expected):", sum(xi * yi for xi, yi in zip(x, y)) + bias)
        print("<x,y> (decrypted):", ip + bias)


if __name__ == "__main__":
    x_input = [0, 133, 254, 9, 205, 248, 126, 254, 182]
    y_input = [-0.03, -0.0261, -0.0194, 0.0786, 0.3495, -0.0135, -0.342, -0.181, -0.2126]

    fhe = BFVFHE()
    fhe.setup(len(x_input), n_length=30)  # Fast prototyping


    fhe.run(x_input, y_input, -0.016008036211133003)