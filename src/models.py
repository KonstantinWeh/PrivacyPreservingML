import torch
import torch.nn as nn
import torch.nn.functional as F
from .altered_ipfe import IPFE
from .optimized_cnn_ipfe import IPFE as OptimizedIPFE
from .optimized_cnn_ipfe import decrypt_patches_batch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import matplotlib.pyplot as plt

# ---- shared builder ----
def build_backbone(cfg):
    c = cfg["model"]
    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(c["in_channels"], c["c1"], kernel_size=c["k1"], padding=1)
            self.bn1   = nn.BatchNorm2d(c["c1"])
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(c["c1"], c["c2"], kernel_size=c["k2"], padding=1)
            self.bn2   = nn.BatchNorm2d(c["c2"])
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Conv2d(c["c2"], c["c3"], kernel_size=c["k3"], padding=1)
            self.bn3   = nn.BatchNorm2d(c["c3"])
            self.pool3 = nn.MaxPool2d(2, 2)

            # for MNIST 28x28 → 14x14 → 7x7 → 3x3 after 3 pools
            self.fc1 = nn.Linear(c["c3"] * 3 * 3, 128)
            self.dropout = nn.Dropout(c.get("dropout", 0.5))
            self.fc2 = nn.Linear(128, c["num_classes"])

        def forward_body(self, x):
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return _Backbone()

class LightweightCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
    @torch.no_grad()
    def load_from_lightweight(self, path_or_state, map_location="cpu"):
        """
        Load state_dict saved from a LightweightCNN checkpoint.
        """
        if isinstance(path_or_state, (str, bytes)):
            src = torch.load(path_or_state, map_location=map_location)
        else:
            src = path_or_state
        # The backbone of LightweightCNN is registered under 'backbone.*'
        if "state_dict" in src:
            state_dict = src["state_dict"]
        else:
            state_dict = src
        # Filter only state_dict keys (could be NestedCheckpoint artifact)
        # Remove "module." prefix if present due to DDP/storage
        cleaned_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            cleaned_dict[k] = v
        # Only load backbone weights
        backbone_weights = {k.replace("backbone.", ""): v for k, v in cleaned_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_weights, strict=False)
        print("Loaded weights into LightweightCNN backbone from checkpoint.")
    def forward(self, x):
        x = x.to(torch.float32)
        return self.backbone.forward_body(x)

class IPFECNN(nn.Module):
    """
    Shares the exact same backbone as LightweightCNN.
    After loading weights from the trained LightweightCNN, it builds IPFE materials
    from conv1 weights and can run the first conv either via IPFE (encrypted=True) or
    via the regular conv1 (encrypted=False).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)   # SAME layers as LightweightCNN

        if cfg["optimizations"]["optimized_ipfe"]:
            self.ipfe = OptimizedIPFE(cfg.get("model", {}).get("prime", 1000000007))
        else:
            self.ipfe = IPFE(cfg.get("model", {}).get("prime", 1000000007))
        

        # --- encryption configuration ---
        k1 = cfg["model"].get("k1", 3)
        self.encryption_length = k1 * k1

        self.ipfe.setup(self.encryption_length)
        print(f"IPFE setup done, with length: {self.encryption_length}")

        # runtime flags (can be driven from cfg)
        mcfg = cfg.get("model", {})
        dbg = mcfg.get("debug_first_conv", {})
        self.debug_compare = bool(dbg.get("compare", False))
        self.debug_center_window = int(dbg.get("center_window", 5))
        self.debug_example_batch = int(dbg.get("example_batch", 0))
        self.optimizations = cfg.get("optimizations", {})
        self.precrypted = bool(self.optimizations.get("precrypted"))
        self.kernel_parallelization = bool(self.optimizations.get("kernel_parallelization"))
        self.kernel_patches_parallelization = bool(self.optimizations.get("kernel_patches_parallelization"))
        self.batch_parallelization = bool(self.optimizations.get("batch_parallelization"))
        self.batch_kernels_parallelization = bool(self.optimizations.get("batch_kernels_parallelization"))
        if sum([self.kernel_parallelization, self.kernel_patches_parallelization, self.batch_parallelization, self.batch_kernels_parallelization]) > 1:
            raise ValueError("Only one of kernel_parallelization, kernel_patches_parallelization, batch_parallelization, batch_kernels_parallelization can be true")
        if sum([self.kernel_parallelization, self.kernel_patches_parallelization, self.batch_parallelization, self.batch_kernels_parallelization]) == 1 and cfg["optimizations"]["optimized_ipfe"] == False:
            raise ValueError("optimized_ipfe must be true if one of kernel_parallelization, kernel_patches_parallelization, batch_parallelization, batch_kernels_parallelization is true")
        
        # prepared after loading weights
        self._ipfe_ready = False
        self.y_array = None
        self.sk_y_array = None
        self.biases = None

    # ---------- weight import from LightweightCNN ----------
    @torch.no_grad()
    def load_from_lightweight(self, path_or_state, map_location="cpu"):
        """
        Load state_dict saved from LightweightCNN (same naming under 'backbone.*').
        Also prepares IPFE materials from conv1 weights.
        """
        if isinstance(path_or_state, (str, bytes)):
            src = torch.load(path_or_state, map_location=map_location)
        else:
            src = path_or_state

        missing, unexpected = self.load_state_dict(src, strict=False)
        print(f"weights copied from trained model (missing={len(missing)}, unexpected={len(unexpected)})")

        self._prepare_ipfe_from_conv1()
        print("sk_ys created")
        print("IPFE ready, successfully loaded weights from lightweightcnn")

    @torch.no_grad()
    def _prepare_ipfe_from_conv1(self):
        # Build IPFE vectors from conv1 weights and biases
        w = self.backbone.conv1.weight.data  # (out_ch, in_ch, k, k)
        self.S_y = 10000
        self.biases = self.backbone.conv1.bias.data
        # flatten each kernel and scale (like your code)
        # NOTE: expects in_ch == 1 for MNIST; if >1, the unfold must match.
        # self.y_array = torch.round(w.view(w.size(0), -1) * 10000).long().tolist()
        self.y_array = torch.round(w.view(w.size(0), -1).squeeze(1).view(w.size(0), -1) * self.S_y).long().tolist()
        print("weights converted to y vectors")
        self.sk_y_array = [self.ipfe.key_derive(y) for y in self.y_array]
        print("sk_y_array created:", self.sk_y_array)
        print("biases saved")
        self._ipfe_ready = True
    
    def encrypt_data(self, x):
        # unfold with same kernel/padding/stride you use for conv1
        x = x.to(torch.float32)
        ksize = self.backbone.conv1.kernel_size[0]
        pad   = self.backbone.conv1.padding[0]
        stride= self.backbone.conv1.stride[0]
        B, _, H, W = x.shape
        print("x shape:", x.shape)


        unfold = nn.Unfold(kernel_size=ksize, stride=stride, padding=pad)
        patches = unfold(x)  # shape: (B, in_ch*ksize*ksize, H*W) with stride/padding applied
        num_patches = patches.shape[-1]

        encrypted_patches = []
        for b in range(B):
            patches_b = patches[b].T  # (num_patches, in_ch*ksize*ksize)
            encrypted_image = []
            for p in range(num_patches):
                patch = patches_b[p]
                patch_int = [int(val.item()) % (self.ipfe.p - 1) for val in patch]

                encrypted  = self.ipfe.encrypt(patch_int)
                encrypted_image.append(encrypted)
            encrypted_patches.append(encrypted_image)
        print("length of encrypted_patches:", len(encrypted_patches))
        print("length of encrypted_patches[0][0]:", len(encrypted_patches[0][0]))
        return encrypted_patches
    

    # ----------------------
    # Helper function for a single patch
    # not yet tested (requires different process kernel)
    # ----------------------
    def process_patch(self, p_idx, patch, sk_y, y_vec, bias):
        val = self.ipfe.decrypt(patch, sk_y, y_vec, max_ip=self.ipfe.p)
        decrypted_val = (val / 10000.0) + bias
        return p_idx, decrypted_val

    def process_kernel_paral_patches(self, k, encrypted_patches):
        """
        Decrypt one kernel across all batches, parallelizing over patches within each batch.

        Returns
        -------
        k : int
            Kernel index.
        results : list[list[float]]
            Outer list has length B (batches), inner lists have length num_patches.
        """
        sk_y = self.sk_y_array[k]
        y_vec = self.y_array[k]
        bias = self.biases[k].item()

        B = len(encrypted_patches)
        num_patches = len(encrypted_patches[0])

        results = []

        # Iterate over batches; for each batch, parallelize across patches
        for b in range(B):
            kernel_results = [0] * num_patches

            # Thread across patches for this batch
            with ThreadPoolExecutor(max_workers=min(num_patches, 8)) as patch_executor:
                patch_futures = [
                    patch_executor.submit(
                        self.process_patch,
                        p_idx,
                        encrypted_patches[b][p_idx],
                        sk_y,
                        y_vec,
                        bias,
                    )
                    for p_idx in range(num_patches)
                ]
                for f in patch_futures:
                    p_idx, val = f.result()
                    kernel_results[p_idx] = val

            results.append(kernel_results)

        return k, results

    # ----------------------
    # Helper function for one kernel
    # ----------------------
    def process_kernel(self, k, encrypted_patches):
        sk_y = int(self.sk_y_array[k])
        y_vec = [int(v) for v in self.y_array[k]]
        bias = self.biases[k].item()
        results = []
        for b in range(len(encrypted_patches)):
            patch_results = []
            for p_idx, patch in enumerate(encrypted_patches[b]):
                val = self.ipfe.decrypt(patch, sk_y, y_vec, max_ip=self.ipfe.p)
                decrypted_val = (val / self.S_y) + bias  # scale + bias
                patch_results.append(decrypted_val)
            results.append(patch_results)
        return k, results

    def decrypt_kernel(self, k, ct0_array, cts_array):
        sk_y = int(self.sk_y_array[k])
        y_vec = np.array(self.y_array[k], dtype=np.int64)
        bias = float(self.biases[k].item())
        decrypted_vals = decrypt_patches_batch(ct0_array, cts_array, sk_y, y_vec, self.ipfe.g, self.ipfe.p)
        return k, decrypted_vals / self.S_y + bias

    # ---------- encrypted first conv ----------
    def first_conv_forward(self, x, precrypted: bool):
        # Store original input shape before encryption
        H, W = 28, 28 # hardcoded bc of MNIST
        
        if not precrypted:
            encrypted_patches = self.encrypt_data(x)  # shape: (B, num_patches, encryption_length + 1)
        else:
            encrypted_patches = x
    

        B = len(encrypted_patches)
        
        ksize = self.backbone.conv1.kernel_size[0]
        pad   = self.backbone.conv1.padding[0]
        stride= self.backbone.conv1.stride[0]

        assert self._ipfe_ready, "Call load_from_lightweight(...) before encrypted forward."
        device = torch.device(self.cfg["device"] if torch.cuda.is_available() else "cpu")
        
        num_patches = len(encrypted_patches[0])
        num_kernels = len(self.sk_y_array)
        
        # Calculate output dimensions
        Hout = int((H + 2 * pad - ksize) / stride + 1)
        Wout = int((W + 2 * pad - ksize) / stride + 1)

        if self.kernel_parallelization:
            # (B, num_kernels, num_patches)
            decrypted_maps = torch.zeros(B, num_kernels, num_patches, device=device)

            # ----------------------
            # Threaded execution across kernels
            # ----------------------
            with ThreadPoolExecutor(max_workers=min(num_kernels, 8)) as executor:
                futures = [
                    executor.submit(self.process_kernel, k, encrypted_patches)
                    for k in range(num_kernels)
                ]
                for f in futures:
                    k, results = f.result()          # results: list of length B
                    for b in range(B):
                        # results[b]: list length num_patches
                        decrypted_maps[b, k, :] = torch.tensor(results[b], device=device)

            # (B, num_kernels, Hout, Wout)
            x_ipfe = decrypted_maps.view(B, num_kernels, Hout, Wout)
        elif self.kernel_patches_parallelization:
            # (B, num_kernels, num_patches)
            decrypted_maps = torch.zeros(B, num_kernels, num_patches, device=device)

            # ----------------------
            # Thread across kernels; inside each kernel, patches are parallelized
            # ----------------------
            with ThreadPoolExecutor(max_workers=min(num_kernels, 8)) as kernel_executor:
                kernel_futures = [
                    kernel_executor.submit(self.process_kernel_paral_patches, k, encrypted_patches)
                    for k in range(num_kernels)
                ]
                for f in kernel_futures:
                    k, results = f.result()  # results: list of length B, each inner list len num_patches
                    for b in range(B):
                        decrypted_maps[b, k, :] = torch.tensor(results[b], device=device)

            # (B, num_kernels, Hout, Wout)
            x_ipfe = decrypted_maps.view(B, num_kernels, Hout, Wout)
        elif self.batch_parallelization:
            # Flatten all batches and patches into a single dimension for numba decryption
            ct0_array = np.array(
                [np.int64(encrypted_patches[b][p][0]) for b in range(B) for p in range(num_patches)]
            )
            cts_array = np.array(
                [np.int64(encrypted_patches[b][p][1]) for b in range(B) for p in range(num_patches)]
            )

            # (B, num_kernels, num_patches)
            decrypted_maps = torch.zeros(B, num_kernels, num_patches, device=device)

            # Loop over kernels
            for k in range(num_kernels):
                sk_y = int(self.sk_y_array[k])
                y_vec = np.array(self.y_array[k], dtype=np.int64)
                bias = float(self.biases[k].item())

                # Batch decrypt all patches using Numba
                decrypted_vals = decrypt_patches_batch(
                    ct0_array, cts_array, sk_y, y_vec, self.ipfe.g, self.ipfe.p
                )

                # Scale, add bias, and reshape to (B, num_patches)
                decrypted_vals_tensor = (
                    torch.tensor(decrypted_vals, device=device) / self.S_y + bias
                ).view(B, num_patches)

                # Store per batch
                for b in range(B):
                    decrypted_maps[b, k, :] = decrypted_vals_tensor[b]

            # (B, num_kernels, Hout, Wout)
            x_ipfe = decrypted_maps.view(B, num_kernels, Hout, Wout)
        elif self.batch_kernels_parallelization:
            # Flatten all batches and patches into a single dimension for numba decryption
            ct0_array = np.array(
                [np.int64(encrypted_patches[b][p][0]) for b in range(B) for p in range(num_patches)]
            )
            cts_array = np.array(
                [np.int64(encrypted_patches[b][p][1]) for b in range(B) for p in range(num_patches)]
            )

            # (B, num_kernels, num_patches)
            decrypted_maps = torch.zeros(B, num_kernels, num_patches, device=device)

            with ThreadPoolExecutor(max_workers=min(num_kernels, os.cpu_count() or 4)) as executor:
                futures = [
                    executor.submit(self.decrypt_kernel, k, ct0_array, cts_array)
                    for k in range(num_kernels)
                ]
                for f in futures:
                    k, vals = f.result()  # flat array length B * num_patches
                    vals_tensor = torch.tensor(vals, device=device).view(B, num_patches)
                    for b in range(B):
                        decrypted_maps[b, k, :] = vals_tensor[b]

            # Reshape to CNN feature map: (B, num_kernels, Hout, Wout)
            x_ipfe = decrypted_maps.view(B, num_kernels, Hout, Wout)


        else:
            # Process each batch
            feature_maps_batch = torch.zeros(B, num_kernels, Hout, Wout, device=device)
            for b in range(B):
                decrypted_maps = torch.zeros(num_kernels, num_patches, device=device)
                
                for k in range(num_kernels):
                    for p in range(num_patches):
                        ct0, ct = encrypted_patches[b][p]
                        decrypted_scaled = self.ipfe.decrypt((ct0, ct), self.sk_y_array[k], self.y_array[k])
                        # (sum(x_i*y_i)*10000)/10000 -> sum(x_i*y_i); add bias
                        decrypted = (decrypted_scaled / (self.S_y)) + self.biases[k].item()
                        decrypted_maps[k, p] = decrypted
                # Reshape to (num_kernels, H_out, W_out) consistent with unfold settings
                feature_maps_b = decrypted_maps.view(num_kernels, Hout, Wout)
                feature_maps_batch[b] = feature_maps_b
            x_ipfe = torch.stack(feature_maps_batch, dim=0)  # (B, num_kernels, Hout, Wout)
        return x_ipfe

    # ---------- full forward ----------
    def forward(self, x):
        x = self.first_conv_forward(x, precrypted=self.precrypted)
        x = self.backbone.pool1(F.relu(self.backbone.bn1(x)))

        # remaining layers are the same as LightweightCNN
        x = self.backbone.pool2(F.relu(self.backbone.bn2(self.backbone.conv2(x))))
        x = self.backbone.pool3(F.relu(self.backbone.bn3(self.backbone.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.backbone.fc1(x))
        x = self.backbone.dropout(x)
        x = self.backbone.fc2(x)
        return x