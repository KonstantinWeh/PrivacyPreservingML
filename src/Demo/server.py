import socket
import json
import random
import numpy as np
from ..math_helper import inv_mod, factor, bsgs, find_generator
from utils.math_helper import inv_mod, bsgs, find_generator, mod_pow_numba, mod_inv_numba, bsgs_numba
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from cryptography.cnn_ipfe import IPFE

HOST = "127.0.0.1"
PORT = 5000

model_path = f"cnn_model.pth"

def handle(command, data):
    """
    command: string
    data:    Python dict parsed from JSON
    """
    if command == "WEIGHTS":
        return {"status": "OK"}

    elif command == "INITIALIZE":
        username = data.get("username")
        age = data.get("age")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ipfe_model = IPFECNN(num_classes=10, prime=4590007).to(device)
        print(f"IPFE-CNN model created on device: {device}")

        return {"status": "created", "user": username, "age": age}

    elif command == "INFERENCE":

        test_ipfe_cnn(ipfe_model, encrypted_data, labels, H, W, device)

        return {"result": a + b}

    return {"error": "unknown command"}


def parse_message(raw):
    """
    Expect format: COMMAND|{json}
    """
    try:
        command, json_str = raw.split("|", 1)
        data = json.loads(json_str)
        return command.strip(), data
    except Exception as e:
        return None, None


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")

        while True:
            raw = conn.recv(4096).decode()
            if not raw:
                break

            command, data = parse_message(raw)
            if command is None:
                conn.sendall(b'{"error":"bad format"}')
                continue

            response = handle(command, data)
            conn.sendall(json.dumps(response).encode())


def test_ipfe_cnn(model, encrypted_data, labels, H, W, device):
    """Test the IPFE-CNN with a sample query vector"""
    model.eval()

    with torch.no_grad():
        print("Testing IPFE-CNN forward pass on encrypted data...")
        print(f"Labels of test samples: {labels.cpu().numpy()}")

        try:
            outputs = model.forward(encrypted_data, encrypted=True, H=28, W=28)
            _, predicted = outputs.max(1)

            print(f"Predictions on encrypted data: {predicted.cpu().numpy()}")

            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            print(f"Accuracy on encrypted samples: {100 * correct / total:.2f}% ({correct}/{total})")


        except Exception as e:
            print(f"Encrypted IPFE forward pass failed: {e}")

class IPFE:
    def __init__(self, l):
        self.p = None
        self.g = None
        self.length = l
        self.mpk = None

    def setup(self, prime, generator, mpk):

        self.p = prime
        self.g = generator
        self.mpk = mpk

    def decrypt(self, ct, sk_y, y):
        ct0, cts = ct

        # prod_i ct_i^{y_i}
        num = 1
        for ci, yi in zip(cts, y):
            num = (num * pow(ci, yi % (self.p - 1), self.p)) % self.p

        # ct0^{sk_y}
        denom = pow(ct0, sk_y % (self.p - 1), self.p)
        val = (num * inv_mod(denom, self.p)) % self.p

        # discrete log base g to recover <x,y>  (works for small message ranges)
        ip = bsgs(self.g, val, self.p)
        if ip is None:
            raise ValueError("discrete log failed (increase prime or reduce message range).")

        modulus = self.p - 1
        half = modulus // 2
        if ip > half:
            ip_signed = ip - modulus
        else:
            ip_signed = ip

        return ip_signed

class IPFECNN(nn.Module):
    def __init__(self, num_classes=10, length):
        super(IPFECNN, self).__init__()
        self.prime = None
        self.ipfe = None
        self.encryption_length = length

        # First convolutional block - this will be used with IPFE
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        #copy weights from the trained model
        self.load_state_dict(torch.load(model_path, map_location=device))
        print("weights copied from trained model")

        self.weights = self.conv1.weight.data
        self.y_array = torch.round(self.weights.view(self.weights.size(0), -1).squeeze(1).view(self.weights.size(0), -1) * 10000).long().tolist()
        print("weights converted to y vectors")
        self.biases = self.conv1.bias
        print("biases saved")
        self.sk_y_array = None

    def setup(self, prime):
        self.ipfe = IPFE(prime)
        self.ipfe.setup(self.encryption_length)
        print("IPFE setup done, with length:", self.encryption_length)

        self.sk_y_array = [self.ipfe.key_derive(y) for y in self.y_array]
        print("sk_ys created")

    def first_conv_forward(self, x):
        num_patches = len(x)
        num_kernels = len(self.sk_y_array)
        device = next(self.parameters()).device

        decrypted_maps = torch.zeros(num_kernels, num_patches, device=device)

        for k in range(num_kernels):
            for p in range(num_patches):
                decrypted_scaled = self.ipfe.decrypt(
                    x[p],
                    self.sk_y_array[k],
                    self.y_array[k],
                )
                decrypted = (decrypted_scaled / 10000) + self.biases[k].item()
                decrypted_maps[k, p] = decrypted
        return torch.stack([decrypted_maps.view(num_kernels, 28, 28)], dim=0)


    def forward(self, x,):
        outputs = []
        for sample in x:
            feat = self.first_conv_forward(sample)
            feat = self.pool1(F.relu(self.bn1(feat)))
            feat = self.pool2(F.relu(self.bn2(self.conv2(feat))))
            feat = self.pool3(F.relu(self.bn3(self.conv3(feat))))
            feat = feat.view(feat.size(0), -1)
            feat = F.relu(self.fc1(feat))
            feat = self.dropout(feat)
            feat = self.fc2(feat)
            outputs.append(feat)
        return torch.cat(outputs, dim=0)



