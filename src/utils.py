import os, json, torch
from pathlib import Path
from datetime import datetime

def make_run_dir(cfg):
    """
    Create (or return) the base directory for this experiment's artifacts.

    With the new layout, we no longer use a per-run name; instead we rely on:
      - cfg["save"]["out_dir"] for results/metrics
      - cfg["save"]["models_dir"] for model checkpoints
    """
    base = Path(cfg["save"]["models_dir"])
    base.mkdir(parents=True, exist_ok=True)
    return base

def save_config(path: Path, cfg: dict):
    with open(path / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)

def _build_model_tag_from_cfg(cfg: dict) -> str:
    """
    Build a descriptive checkpoint tag from the config.

    Example (for MNIST defaults):
        k3_conv16_32_64_stride1_pad1_dropout0.5
    """
    model_cfg = cfg.get("model", {})

    # Kernel sizes and channels
    k_list = model_cfg.get("k", [])
    c_list = model_cfg.get("c", [])
    stride_list = model_cfg.get("stride", [])
    pad_list = model_cfg.get("padding", [])
    dropout = model_cfg.get("dropout", None)

    # Use first kernel/stride/pad (common case: same for all convs)
    k_first = k_list[0] if isinstance(k_list, (list, tuple)) and len(k_list) > 0 else "unknown"
    stride_first = stride_list[0] if isinstance(stride_list, (list, tuple)) and len(stride_list) > 0 else "unknown"
    pad_first = pad_list[0] if isinstance(pad_list, (list, tuple)) and len(pad_list) > 0 else "unknown"

    # All conv channels
    if isinstance(c_list, (list, tuple)) and len(c_list) > 0:
        conv_str = "_".join(str(c) for c in c_list)
    else:
        conv_str = "unknown"

    tag = f"k{k_first}_conv{conv_str}_stride{stride_first}_pad{pad_first}"
    if dropout is not None:
        tag += f"_dropout{dropout}"
    return tag


def save_checkpoint(model, cfg):
    """
    Save model checkpoint.
    """
    tag = _build_model_tag_from_cfg(cfg)
    Path(cfg["save"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(cfg["save"]["models_dir"]) / f"{tag}.pt")

def find_checkpoint(cfg: dict = None) -> Path:
    """
    Find checkpoint file in a directory.
    
    Args:
        cfg: Config dict to generate checkpoint name
    """
    tag = _build_model_tag_from_cfg(cfg)
    checkpoint_path = Path(cfg["save"]["models_dir"]) / f"{tag}.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
