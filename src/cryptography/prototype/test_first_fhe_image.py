import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import random
from src.cryptography.cnn_fhe import FHE

if __name__ == "__main__":
    class ToTensor255:
        def __call__(self, pic):
            # Convert PIL image to a torch tensor (H x W)
            return torch.tensor(np.array(pic), dtype=torch.float32)


    transform = transforms.Compose([
        ToTensor255()
    ])

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
    image_array = image.view(-1).cpu().numpy()
    print(f"Flattened image shape: {image_array.shape}")

    # For FHE we keep values as floats (no mod p-1 reduction needed)
    x = image_array.tolist()  # full 28x28 = 784 values as floats

    plt.imshow(np.asarray(x).reshape(28, 28), cmap='gray')
    plt.title("Plaintext flattened image (floats)")
    plt.show()
    plt.close()

    # Random vector y (same length as flattened image)
    y = [random.uniform(-1, 1) for _ in range(len(x))]  # random floats in [-1,1]
    print(f"Vector lengths: x={len(x)}, y={len(y)}")
    print(f"x[0]={x[0]}, type={type(x[0])}")
    print(f"y[0]={y[0]}, type={type(y[0])}")

    # FHE demo
    fhe_demo = FHE()
    fhe_demo.setup(len(x), poly_modulus_degree=4096, coeff_mod_bit_sizes=[40, 20, 40], scale_bits=20)
    fhe_demo.run(x, y)
