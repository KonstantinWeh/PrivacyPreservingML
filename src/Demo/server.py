import socket
import json
from ..math_helper import inv_mod, bsgs
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct

HOST = "127.0.0.1"
PORT = 5000

class IPFE:
    def __init__(self, prime, generator, l):
        self.p = prime
        self.g = generator
        self.length = l

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
    def __init__(self, device, num_classes=10):
        super(IPFECNN, self).__init__()
        self.prime = None
        self.ipfe = None

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
        self.load_state_dict(torch.load("src/Demo/model.pth", map_location=device))
        print("weights copied from trained model")

        self.weights = self.conv1.weight.data
        self.y_array = torch.round(self.weights.view(self.weights.size(0), -1).squeeze(1).view(self.weights.size(0), -1) * 10000).long().tolist()
        print("weights converted to y vectors")
        self.biases = self.conv1.bias
        print("biases saved")
        self.sk_y_array = None

    def setup(self, prime, generator, length, sk_y):
        self.ipfe = IPFE(prime, generator, length)
        self.sk_y_array = sk_y
        print("Ipfe setup done and sk_ys saved")

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_model = IPFECNN(device=device, num_classes=10)
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
        data_set = data.get("dataset")

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

