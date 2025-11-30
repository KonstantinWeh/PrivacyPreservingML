import socket
from ..altered_ipfe import IPFE

HOST = "127.0.0.1"
PORT = 5000

with ((socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s)):
    s.connect((HOST, PORT))
    print("Connected to server.")

    ipfe = None
    encrypted_data = None
    prime = 4590007
    length = 9

    while True:
        command = input("Enter a Command: 0: quit, 1: setup, 2: encrypt, 3: request inference")

        if command == "0":
            break

        elif command == "1":
            print("Initializing IPFE...")
            # send length and receive weights

            weights = None
            # use weights to initialize IPFE
            ipfe = IPFE(prime)
            ipfe.setup(length)
            num_kernel = len(weights)
            sk_y = []
            for i in range(num_kernel):
                sk_y[i] = ipfe.key_derive(weights[i])

            p = ipfe.p
            g = ipfe.g
            mpk = ipfe.mpk

            # Send prime, generator, mpk, sk_y


            s.sendall("msg".encode())
            data = s.recv(1024)
            print("Server replied:", data.decode())

        elif command == "2":
            if ipfe is None:
                print("IPFE not initialized.")
            else:

                s.sendall("msg".encode())
                data = s.recv(1024)
                print("Server replied:", data.decode())

        elif command == "3":
            if ipfe is None:
                print("IPFE not initialized.")
            elif encrypted_data is None:
                print("Data not encrypted.")
            else:

                s.sendall("msg".encode())
                data = s.recv(1024)
                print("Server replied:", data.decode())