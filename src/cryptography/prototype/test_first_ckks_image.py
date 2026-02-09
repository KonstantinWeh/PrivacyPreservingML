import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import random

from src.cryptography.cnn_ckks import FHE  # your FHE class


if __name__ == "__main__":
    class ToTensor255:
        def __call__(self, pic):
            # Convert PIL image to a torch tensor (H x W)
            return torch.tensor(np.array(pic), dtype=torch.float32)

    transform = transforms.Compose([ToTensor255()])

    # Load MNIST dataset
    train_dataset = MNIST(root='../../../data', train=True, download=True, transform=transform)

    # Get first image
    image = train_dataset[0][0]
    label = train_dataset[0][1]

    plt.imshow(image, cmap='gray')
    plt.title(f"Original image (label: {label})")
    plt.show()
    plt.close()
    print(f"Image shape: {image.shape}, Label: {label}")

    # Flatten image
    image_flat = image.view(-1).cpu().numpy()  # (784,)
    print(f"Flattened image shape: {image_flat.shape}")

    x = image_flat.tolist()

    # Show flattened as image again (sanity check)
    plt.imshow(np.asarray(x).reshape(28, 28), cmap='gray')
    plt.title("Plaintext flattened image (floats)")
    plt.show()
    plt.close()

    # --- FHE setup and encryption of full image vector ---
    fhe_demo = FHE()
    fhe_demo.setup(
        len(x),
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[40, 20, 40],
        scale_bits=20
    )

    # Encrypt the entire image vector
    ct_image = fhe_demo.encrypt(x)
    print("Encrypted image type:", type(ct_image))
    print("Encrypted image repr (truncated):", str(ct_image)[:200], "...")

    # --- Decrypt and visualize reconstructed image ---
    decrypted_list = ct_image.decrypt()  # list of floats, length 784
    decrypted_arr = np.array(decrypted_list).reshape(28, 28)

    plt.imshow(decrypted_arr, cmap='gray')
    plt.title("Decrypted image (CKKS approximation)")
    plt.show()
    plt.close()

    # --- Quantitative check: difference between original and decrypted ---
    orig_arr = np.array(x).reshape(28, 28)
    diff = decrypted_arr - orig_arr
    print("Max abs difference:", np.max(np.abs(diff)))
    print("Mean abs difference:", np.mean(np.abs(diff)))
