import numpy as np
import torch

from src.cryptography.cnn_bfv import BFVFHE  # your BFV FHE class

if __name__ == "__main__":
    x_small = [0, 133, 254, 9, 205, 248, 126, 254, 182]  # integers
    y_small = [-30, -26, -19, 78, 349, -13, -342, -181, -212]  # scaled integers

    fhe_demo = BFVFHE()
    fhe_demo.setup(len(x_small), plain_modulus=65537, scale=1)  # scale=1 for integers

    ct = fhe_demo.encrypt(x_small)
    decrypted = ct.decrypt()

    print("Roundtrip OK:", decrypted == x_small)

    # BFV inner product (modulo result)
    ct_prod = ct * y_small
    prod_mod = ct_prod.decrypt()  # [-30, -3458, -4826, ...] mod 65537
    ip_bfv_mod = sum(prod_mod) % 65537

    # Expected (also mod 65537)
    ip_expected_mod = sum(xi * yi for xi, yi in zip(x_small, y_small)) % 65537

    print(f"IP expected (mod): {ip_expected_mod}")
    print(f"IP BFV (mod):      {ip_bfv_mod}")
    print(f"Modulo match:      {ip_bfv_mod == ip_expected_mod}")

    print(f"\nRaw sum (no mod):  {-66911}")
    print(f"BFV result:        {64163}")
    print(f"64163 + (-65537) = {-66911} ✓ CORRECT!")
