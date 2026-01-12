import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np

def encrypt_test_data(model, test_loader, device, num_samples=5):
    """Encrypt a batch of test data"""
    model.eval()
    with torch.no_grad():

        # Get a batch of test data
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        # Encrypt only a subset
        images_subset = images[:num_samples]
        labels_subset = labels[:num_samples]

        # Encrypt the data
        encrypted_data = model.encrypt_data(images_subset)
        print(f"Encrypted {num_samples} samples.")

    return encrypted_data, labels_subset

def test_ipfe_cnn(model, encrypted_data, labels, H, W, device):
    """Test the IPFE-CNN with a sample query vector"""
    model.eval()

    with torch.no_grad():
        print("Testing IPFE-CNN forward pass on encrypted data...")
        print(f"Labels of test samples: {labels.cpu().numpy()}")

        try:
            outputs = model.forward(encrypted_data, encrypted=True, H=H, W=W)
            _, predicted = outputs.max(1)

            print(f"Predictions on encrypted data: {predicted.cpu().numpy()}")

            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            print(f"Accuracy on encrypted samples: {100 * correct / total:.2f}% ({correct}/{total})")


        except Exception as e:
            print(f"Encrypted IPFE forward pass failed: {e}")

def test_regular_ipfe_cnn(model, test_loader, device, num_samples=5):
    """Test the IPFE-CNN with a sample query vector"""
    model.eval()

    with torch.no_grad():
        print("Testing IPFE-CNN forward pass on encrypted data...")

        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        # Encrypt only a subset
        images_subset = images[:num_samples]
        labels_subset = labels[:num_samples]
        try:
            outputs = model.forward(images_subset, encrypted=False, H=28, W=28)
            _, predicted = outputs.max(1)

            print(f"Predictions on encrypted data: {predicted.cpu().numpy()}")

            correct = (predicted == labels_subset).sum().item()
            total = labels_subset.size(0)
            print(f"Accuracy on encrypted samples: {100 * correct / total:.2f}% ({correct}/{total})")


        except Exception as e:
            print(f"Encrypted IPFE forward pass failed: {e}")


def load_data():
    transform = transforms.Compose([
        transforms.Lambda(lambda pic: torch.tensor(np.array(pic), dtype=torch.float32).unsqueeze(0))
    ])
    batch_size = 64
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    return test_loader