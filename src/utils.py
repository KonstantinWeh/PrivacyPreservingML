import os, json, torch
from pathlib import Path
from datetime import datetime

def make_run_dir(cfg):
    base = Path(cfg["save"]["out_dir"])
    name = cfg["save"]["run_name"]
    path = base / f"{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_config(path: Path, cfg: dict):
    with open(path / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)

def save_checkpoint(path: Path, model, tag=None, cfg=None):
    """
    Save model checkpoint.
    
    Args:
        path: Directory to save the checkpoint
        model: Model to save
        tag: Optional tag name for the file. If None, will generate from cfg:
             - Format: {model_name}_k1-{k_first}_conv{num_layers}
             - Example: light_k1-3_conv3.pt
        cfg: Optional config dict to extract model name and architecture params
    """
    if tag is None:
        if cfg is not None and "model" in cfg:
            model_cfg = cfg["model"]
            # Get model name
            model_name = model_cfg.get("name", model.__class__.__name__.lower())
            
            # Get first kernel size from list k
            k_list = model_cfg.get("k", None)
            k_first = k_list[0] if isinstance(k_list, (list, tuple)) and len(k_list) > 0 else None
            
            # Number of convolution layers from length of c list
            c_list = model_cfg.get("c", None)
            num_conv_layers = len(c_list) if isinstance(c_list, (list, tuple)) else 0
            
            # Build tag: {model_name}_k1-{k_first}_conv{num_layers}
            if k_first is not None and num_conv_layers > 0:
                tag = f"{model_name}_k1-{k_first}_conv{num_conv_layers}"
            elif model_name:
                tag = model_name
            else:
                tag = model.__class__.__name__.lower()
        else:
            # Fallback to class name
            tag = model.__class__.__name__.lower()
    
    torch.save(model.state_dict(), path / f"{tag}.pt")

def find_checkpoint(checkpoint_dir: Path, model_name: str = None, cfg: dict = None) -> Path:
    """
    Find checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        model_name: Optional model name or full checkpoint name (e.g., "light", "light_k1-3_conv3"). 
                    If None and cfg is provided, will generate name from cfg.
        cfg: Optional config dict to generate checkpoint name (k1 and conv layers)
    
    Returns:
        Path to the checkpoint file
        
    Raises:
        FileNotFoundError: If no matching checkpoint is found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if model_name is None and cfg is not None:
        # Generate name from cfg using same logic as save_checkpoint
        if "model" in cfg:
            model_cfg = cfg["model"]
            base_name = model_cfg.get("name", "model")
            k_list = model_cfg.get("k", None)
            k_first = k_list[0] if isinstance(k_list, (list, tuple)) and len(k_list) > 0 else None
            c_list = model_cfg.get("c", None)
            num_conv_layers = len(c_list) if isinstance(c_list, (list, tuple)) else 0
            if k_first is not None and num_conv_layers > 0:
                model_name = f"{base_name}_k1-{k_first}_conv{num_conv_layers}"
            else:
                model_name = base_name
    
    if model_name is not None:
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Find any .pt file in the directory
        pt_files = list(checkpoint_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        if len(pt_files) > 1:
            raise ValueError(f"Multiple checkpoint files found in {checkpoint_dir}. Specify model_name or cfg.")
        return pt_files[0]
