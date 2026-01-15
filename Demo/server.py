import socket
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import numpy as np
from numba import njit
from src.utils.math_helper import mod_pow_numba, mod_inv_numba, bsgs_numba

HOST = "127.0.0.1"
PORT = 5000

def main():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        current_model = None
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")

            while True:
                msg = receive_message(conn)
                if msg is None:
                    print("Client disconnected.")
                    break

                print("FULL MESSAGE RECEIVED:")

                command, data = parse_message(msg)
                model, response = handle(current_model, command, data)
                current_model = model

                send_message(conn, response)

@njit
def decrypt_fast(ct0, cts, sk_y, y, p):
    num = 1
    for i in range(len(cts)):
        ci = cts[i]
        yi = y[i]
        num = (num * mod_pow_numba(ci, yi % (p - 1), p)) % p
    denom = mod_pow_numba(ct0, sk_y % (p - 1), p)
    val = (num * mod_inv_numba(denom, p)) % p
    return val

@njit
def decrypt_patches_batch(ct0_array, cts_array, sk_y, y_vec, g, p):
    """
    Fully Numba-decrypted patches with BSGS.
    patches_ct: list of (ct0, cts) tuples
    Returns list of signed inner products.
    """
    num_patches = len(ct0_array)
    results = np.zeros(num_patches, dtype=np.int64)

    y_vec_int = [int(y) for y in y_vec]

    for i in range(num_patches):
        ct0 = int(ct0_array[i])
        cts = [int(c) for c in cts_array[i]]

        # Step 1: modular arithmetic
        val = decrypt_fast(ct0, cts, int(sk_y), y_vec_int, p)

        # Step 2: discrete log
        ip = bsgs_numba(g, val, p, p - 1)
        if ip == -1:
            raise ValueError("Discrete log failed")

        # Step 3: signed adjustment
        modulus = p - 1
        half = modulus // 2
        if ip > half:
            ip_signed = ip - modulus
        else:
            ip_signed = ip

        results[i] = ip_signed

    return results

class IPFE:
    def __init__(self, prime, generator, l):
        self.p = prime
        self.g = generator
        self.length = l

class IPFECNN(nn.Module):
    def __init__(self, version, device, num_classes=10):
        super(IPFECNN, self).__init__()
        self.prime = None
        self.ipfe = None
        self.version = version
        if version == 1:
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

            self.dim_first = 28

        elif version == 2:
            # First convolutional block - this will be used with IPFE
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)

            # Second convolutional block
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Third convolutional block
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(32)
            self.pool3 = nn.MaxPool2d(2, 2)

            self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool4 = nn.MaxPool2d(2, 2)

            # Fully connected layers
            self.fc1 = nn.Linear(64 * 1 * 1, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)

            self.dim_first = 9
        else:
            # First convolutional block - this will be used with IPFE
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=3, padding=1)  # stride = 2, padding = 0
            self.bn1 = nn.BatchNorm2d(8)
            self.pool1 = nn.MaxPool2d(2, 2)

            # Second convolutional block
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Third convolutional block
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(32)
            self.pool3 = nn.MaxPool2d(2, 2)

            # Fully connected layers
            self.fc1 = nn.Linear(32 * 1 * 1, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)

            self.dim_first = 10

        #copy weights from the trained model
        self.load_state_dict(torch.load(f"models/demo_model_{version}.pth", map_location=device))
        print("weights copied from trained model")

        self.weights = self.conv1.weight.data
        self.y_array = torch.round(self.weights.view(self.weights.size(0), -1).squeeze(1).view(self.weights.size(0), -1) * 10000).long().tolist()
        print("weights converted to y vectors")
        self.biases = self.conv1.bias
        print("biases saved")
        self.sk_y_array = None

    def setup(self, prime, generator, length, sk_y):
        self.prime = prime
        self.ipfe = IPFE(prime, generator, length)
        self.sk_y_array = sk_y
        print("IPFE setup complete")

    def first_conv_forward(self, x):
        num_kernels = len(self.sk_y_array)
        device = next(self.parameters()).device

        ct0_array, cts_array = x
        num_patches = ct0_array.shape[0]

        decrypted_maps = torch.zeros(num_kernels, num_patches, device=device)

        # Loop over kernels
        for k in range(num_kernels):
            sk_y = int(self.sk_y_array[k])
            y_vec = np.array(self.y_array[k], dtype=np.int64)
            bias = float(self.biases[k].item())

            # Batch decrypt all patches using Numba
            decrypted_vals = decrypt_patches_batch(ct0_array, cts_array, sk_y, y_vec, self.ipfe.g, self.ipfe.p)

            # Scale and add bias
            decrypted_maps[k, :] = torch.tensor(decrypted_vals / 10000.0 + bias, device=device)

        return decrypted_maps.view(1, num_kernels, self.dim_first, self.dim_first)

    def _forward_tail_v1_3(self, feat):
        feat = self.pool1(F.relu(self.bn1(feat)))
        feat = self.pool2(F.relu(self.bn2(self.conv2(feat))))
        feat = self.pool3(F.relu(self.bn3(self.conv3(feat))))
        return feat

    def _forward_tail_v2(self, feat):
        feat = F.relu(self.bn1(feat))
        feat = self.pool2(F.relu(self.bn2(self.conv2(feat))))
        feat = self.pool3(F.relu(self.bn3(self.conv3(feat))))
        feat = self.pool4(F.relu(self.bn4(self.conv4(feat))))
        return feat

    def forward(self, x):
        outputs = []
        for sample in x:
            feat = self.first_conv_forward(sample)
            if self.version == 2:
                feat = self._forward_tail_v2(feat)
            else:
                feat = self._forward_tail_v1_3(feat)
            feat = feat.view(feat.size(0), -1)
            feat = F.relu(self.fc1(feat))
            feat = self.dropout(feat)
            outputs.append(self.fc2(feat))
        return torch.cat(outputs, dim=0)

def recvall(conn, n):
    """Receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_message(conn):
    """Receive one length-prefixed message."""
    header = recvall(conn, 4)
    if not header:
        return None

    (length,) = struct.unpack("!I", header)
    data = recvall(conn, length)
    if data is None:
        return None

    return data.decode("utf-8")

def send_message(conn, obj):
    """Send Python dict or string back to the client."""
    # Convert Python reply (dict, list, string) to JSON text
    if not isinstance(obj, (str, bytes)):
        obj = json.dumps(obj)
    if isinstance(obj, str):
        obj = obj.encode("utf-8")

    header = struct.pack("!I", len(obj))
    conn.sendall(header)
    conn.sendall(obj)

def parse_message(raw):
    """
    Expect format: COMMAND|{json}
    """
    print(f"Received raw message: {raw}")
    try:
        command, json_str = raw.split("|", 1)
        data = json.loads(json_str)
        return command.strip(), data
    except Exception as e:
        return None, None

def handle(model, command, data):
    if command == "WEIGHTS":
        m = int(data.get("model"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if m == 1:
            new_model = IPFECNN(device=device, version=1, num_classes=10)
        elif m == 2:
            new_model = IPFECNN(device=device, version=2, num_classes=10)
        else:
            new_model = IPFECNN(device=device, version=3, num_classes=10)
        weights = new_model.y_array
        return new_model, {"weights": weights}

    elif command == "INITIALIZE":
        p = data.get("prime")
        g = data.get("generator")
        l = data.get("length")
        sk_y = data.get("sk_y")
        model.setup(prime=p, generator=g, length=l, sk_y=sk_y)
        return model, {"status": "initialized"}

    elif command == "INFERENCE":
        raw_data_set = data.get("dataset")

        data_set = [
            (np.array(ct0_list, dtype=np.int64),
             np.array(cts_list, dtype=np.int64))
            for ct0_list, cts_list in raw_data_set
        ]

        model.eval()

        with torch.no_grad():
            print("IPFE-CNN forward pass on encrypted data...")

            try:
                outputs = model.forward(data_set)
                _, predicted = outputs.max(1)

                print(f"Predictions on encrypted data: {predicted.cpu().numpy()}")

                return model, {"predictions": predicted.cpu().numpy().tolist()}

            except Exception as e:
                print(f"Encrypted IPFE forward pass failed: {e}")
        return model, {"error": "inference failed"}

    return model, {"error": "unknown command"}

if __name__ == "__main__":
    main()
