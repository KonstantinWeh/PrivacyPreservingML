import socket
import json
import random
from ..math_helper import find_generator
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
import struct

HOST = "127.0.0.1"
PORT = 5000

prime = 4590007
length = 9

class IPFE:
    def __init__(self, p):
        self.p = p
        self.g = None
        self.length = None
        self.mpk = None
        self.msk = None

    def setup(self, l):
        self.length = l
        self.g = find_generator(self.p)
        s = [random.randrange(1, self.p - 1) for _ in range(self.length)]
        h = [pow(self.g, s_i, self.p) for s_i in s]

        self.mpk = h
        self.msk = s

    def encrypt(self, x):
        if len(x) != self.length:
            raise ValueError("x length does not match setup length.")

        r = random.randrange(1, self.p - 1)

        ct0 = pow(self.g, r, self.p)

        ct = [(pow(h_i, r, self.p) * pow(self.g, x_i % (self.p - 1), self.p)) % self.p for h_i, x_i in zip(self.mpk, x)]

        return ct0, ct

    def key_derive(self, y):
        if len(y) != self.length:
            raise ValueError("y length does not match setup length.")
        return sum((si * yi) % (self.p - 1) for si, yi in zip(self.msk, y)) % (self.p - 1)

def send_message(sock, text):
    data = text.encode("utf-8")
    length = len(data)
    sock.sendall(struct.pack("!I", length))  # send 4-byte length
    sock.sendall(data)

def recvall(sock, n):
    """Receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_message(sock):
    """Receive a length-prefixed message."""
    # Read 4-byte header
    header = recvall(sock, 4)
    if not header:
        return None

    (length,) = struct.unpack("!I", header)

    # Read the full message
    data = recvall(sock, length)
    return data.decode("utf-8")

def send(sock, command, data):
    message = f"{command}|{json.dumps(data)}"
    message_bytes = message.encode("utf-8")

    header = struct.pack("!I", len(message_bytes))
    sock.sendall(header)

    sock.sendall(message_bytes)

    reply = receive_message(sock)
    return json.loads(reply)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to server.")

    ipfe = None
    encrypted_data = None

    while True:
        command = input("0: quit, 1: setup, 2: encrypt, 3: request inference \n Enter a Command: ")

        if command == "0":
            break

        elif command == "1":
            print("Request weights from server...")

            data = send(s, "WEIGHTS", {})
            y_array = data.get("weights")
            print("Received weights:")

            print("Initializing IPFE...")
            ipfe = IPFE(prime)
            ipfe.setup(length)
            generator = ipfe.g
            sk_y_array = [ipfe.key_derive(y) for y in y_array]

            json_data = {
                "prime": prime,
                "generator": generator,
                "length": length,
                "sk_y": sk_y_array
            }

            print("Sending ipfe data...")
            data = send(s, "INITIALIZE", json_data)

            print("Server replied:", data)

        elif command == "2":
            if ipfe is None:
                print("IPFE not initialized.")
            else:
                print("Encrypting data...")
                transform = transforms.Compose([
                    transforms.Lambda(lambda pic: torch.tensor(np.array(pic), dtype=torch.float32).unsqueeze(0))
                ])
                batch_size = 64
                test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    data_iter = iter(test_loader)
                    images, labels = next(data_iter)
                    images = images.to(device)

                    images_subset = images[:5]
                    labels_subset = labels[:5]

                    unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
                    patches = unfold(images_subset)
                    B, patch_size, num_patches = patches.shape

                    encrypted_patches = []

                    for b in range(B):
                        patches_b = patches[b].T
                        encrypted_image = []
                        for p in range(num_patches):
                            patch = patches_b[p]
                            patch_int = [(int(val.item()) % (prime - 1)) for val in patch]
                            encrypted = ipfe.encrypt(patch_int)
                            encrypted_image.append(encrypted)
                        encrypted_patches.append(encrypted_image)

                    encrypted_data = encrypted_patches
                    print(f"Encrypted 5 samples.")
                    print("Labels for the encrypted samples:", labels_subset.numpy())


        elif command == "3":
            if ipfe is None:
                print("IPFE not initialized.")
            elif encrypted_data is None:
                print("Data not encrypted.")
            else:
                print("Requesting inference from server...")

                data = send(s, "INFERENCE", {"dataset": encrypted_data})

                print("Server replied:", data)
