import numpy as np
from src.cryptography.prototype.first_ipfe import IPFE
import random
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import torch

# initial test run of the IPFE scheme with image inputs

if __name__ == "__main__":
    class ToTensor255:
        def __call__(self, pic):
            # Convert PIL image to a torch tensor (H x W)
            return torch.tensor(np.array(pic), dtype=torch.float32)


    transform = transforms.Compose([
        ToTensor255()
    ])

    # Load training and test datasets
    train_dataset = MNIST(root='../../../data', train=True, download=True, transform=transform)

    print(train_dataset[0])

    image = train_dataset[0][0]
    label = train_dataset[0][1]
    plt.imshow(image, cmap='gray')
    plt.title("Original image")
    plt.show()
    plt.close()
    print(f"Image shape: {image.shape}, Label: {label}")

    image_array = image.view(-1).cpu().numpy()# [int(val * 1000) % (self.prime - 1) for val in x_flat[i]]
    print(f"Flattened image shape: {image_array.shape}")

    p = 104729
    x = [int(val) % (p - 1) for val in image_array]

    plt.imshow(np.asarray(x).reshape(28, 28), cmap='gray')
    plt.title("Converted image")
    plt.show()
    plt.close()
    # random vector
    y = [random.randint(0, p - 1) for _ in range(len(x))]
    print(len(x), len(y))
    print(type(x[0]), type(y[0]))

    fe_demo = IPFE(p)
    fe_demo.run(len(x), x, y)