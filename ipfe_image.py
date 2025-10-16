import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from ip_functional_encryption import IPFE
import random

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    # Load training and test datasets
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    image = train_dataset[0][0]
    label = train_dataset[0][1]
    print(f"Image shape: {image.shape}, Label: {label}")
    print(image)

    image_array = image.view(-1).cpu().numpy()# [int(val * 1000) % (self.prime - 1) for val in x_flat[i]]
    print(f"Flattened image shape: {image_array.shape}")

    p = 104729
    x = [int(val * 1000) % (p - 1) for val in image_array]
    # random vector
    y = [random.randint(0, p - 1) for _ in range(len(x))]
    print(len(x), len(y))
    print(type(x[0]), type(y[0]))

    fe_demo = IPFE(p)
    fe_demo.run(len(x), x, y)