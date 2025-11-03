import torch
import torch.nn as nn
import torch.nn.functional as F
from ip_functional_encryption import IPFE

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
        return self.backbone.forward_body(x)

class IPFECNN(nn.Module):
    """
    Shares the exact same backbone as LightweightCNN.
    After loading weights from the trained LightweightCNN, it builds IPFE materials
    from conv1 weights and can run the first conv either via IPFE (encrypted=True) or
    via the regular conv1 (encrypted=False).
    """
    def __init__(self, cfg, ipfe_impl=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)   # SAME layers as LightweightCNN
        self.ipfe = ipfe_impl if ipfe_impl is not None else IPFE(cfg.get("model", {}).get("prime", 1000000007))
        

        # --- encryption configuration ---
        k1 = cfg["model"].get("k1", 3)
        self.encryption_length = k1 * k1

        self.ipfe.setup(self.encryption_length)
        print(f"IPFE setup done, with length: {self.encryption_length}")

        # runtime flags (can be driven from cfg)
        mcfg = cfg.get("model", {})
        self.default_encrypted = bool(mcfg.get("encrypted_default", False))
        dbg = mcfg.get("debug_first_conv", {})
        self.debug_compare = bool(dbg.get("compare", False))
        self.debug_center_window = int(dbg.get("center_window", 5))
        self.debug_example_batch = int(dbg.get("example_batch", 0))

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
        self.biases = self.backbone.conv1.bias.data
        # flatten each kernel and scale (like your code)
        # NOTE: expects in_ch == 1 for MNIST; if >1, the unfold must match.
        self.y_array = torch.round(w.view(w.size(0), -1) * 10000).long().tolist()
        print("weights converted to y vectors")
        self.sk_y_array = [self.ipfe.key_derive(y) for y in self.y_array]
        print("biases saved")
        self._ipfe_ready = True

    # ---------- encrypted first conv ----------
    def first_conv_forward(self, x, encrypted: bool):
        if not encrypted:
            return self.backbone.conv1(x)

        assert self._ipfe_ready, "Call load_from_lightweight(...) before encrypted forward."
        device = x.device
        B, _, H, W = x.shape

        # unfold with same kernel/padding/stride you use for conv1
        ksize = self.backbone.conv1.kernel_size[0]
        pad   = self.backbone.conv1.padding[0]
        stride= self.backbone.conv1.stride[0]

        unfold = nn.Unfold(kernel_size=ksize, stride=stride, padding=pad)
        patches = unfold(x)  # shape: (B, in_ch*ksize*ksize, H*W) with stride/padding applied

        num_patches = patches.shape[-1]
        num_kernels = len(self.sk_y_array)

        feature_maps_batch = []

        # ---------- Logging like your snippet ----------
        print(f"[IPFE] encrypted first conv | B={B} k={ksize}x{ksize} stride={stride} pad={pad} "
              f"| kernels={num_kernels} | patches/feature={num_patches}")

        for b in range(B):
            patches_b = patches[b].T  # (num_patches, in_ch*ksize*ksize)
            decrypted_maps = torch.zeros(num_kernels, num_patches, device=device)

            print(f"Decrypting {num_kernels} kernels with {num_patches} patches")

            for p in range(num_patches):
                patch = patches_b[p]
                patch_int = [int(val.item()) % (self.ipfe.p - 1) for val in patch]
                ct = self.ipfe.encrypt(patch_int)

                for k in range(num_kernels):
                    decrypted_scaled = self.ipfe.decrypt(ct, self.sk_y_array[k], self.y_array[k])
                    # (sum(x_i*y_i)*10000)/10000 -> sum(x_i*y_i); add bias
                    decrypted = (decrypted_scaled / 10000) + self.biases[k].item()
                    decrypted_maps[k, p] = decrypted
            # Reshape to (num_kernels, H_out, W_out) consistent with unfold settings
            Hout = int((H + 2*pad - ksize) / stride + 1)
            Wout = int((W + 2*pad - ksize) / stride + 1)
            feature_maps_b = decrypted_maps.view(num_kernels, Hout, Wout)
            feature_maps_batch.append(feature_maps_b)
            print(f"Feature map {b}: {feature_maps_b.shape}")

        x_ipfe = torch.stack(feature_maps_batch, dim=0)  # (B, num_kernels, Hout, Wout)

        # ---------- Optional debug comparison with regular conv ----------
        if self.debug_compare:
            with torch.no_grad():
                x_conv = self.backbone.conv1(x)
                diff = (x_conv - x_ipfe).abs()
                b = min(self.debug_example_batch, B - 1)
                w = max(1, self.debug_center_window)
                for c in range(min(num_kernels, 3)):  # print first 3 kernels to avoid spam
                    Hc, Wc = x_conv.shape[2], x_conv.shape[3]
                    h0, w0 = Hc // 2 - w//2, Wc // 2 - w//2
                    h1, w1 = h0 + w, w0 + w
                    conv_center = x_conv[b, c, h0:h1, w0:w1].detach().cpu()
                    ipfe_center = x_ipfe[b, c, h0:h1, w0:w1].detach().cpu()
                    diff_center = diff[b, c, h0:h1, w0:w1].detach().cpu()
                    print(f"\n=== Kernel {c} — Center {w}×{w} Region (batch {b}) ===")
                    print("Conv Output:\n", conv_center)
                    print("IPFE Output:\n", ipfe_center)
                    print("|Difference|:\n", diff_center)

        return x_ipfe

    # ---------- full forward ----------
    def forward(self, x, encrypted: bool = True):
        """
        encrypted: if None, uses cfg.model.encrypted_default; else overrides per-call.
        """
        if encrypted is None:
            encrypted = self.default_encrypted

        x = self.first_conv_forward(x, encrypted=encrypted)
        x = self.backbone.pool1(F.relu(self.backbone.bn1(x)))

        # remaining layers are the same as LightweightCNN
        x = self.backbone.pool2(F.relu(self.backbone.bn2(self.backbone.conv2(x))))
        x = self.backbone.pool3(F.relu(self.backbone.bn3(self.backbone.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.backbone.fc1(x))
        x = self.backbone.dropout(x)
        x = self.backbone.fc2(x)
        return x