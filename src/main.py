import argparse, yaml, torch
from pathlib import Path
import os

from .seed import set_seed
from .data import make_mnist_loaders
from .models import PlainCNN, IPFECNN
from .train import fit
from .eval import evaluate_top1
from .utils import make_run_dir, save_config, save_checkpoint, find_checkpoint, _build_model_tag_from_cfg

def load_cfg(path_list):
    cfg = {}
    for p in path_list:
        with open(p) as f:
            part = yaml.safe_load(f)
        # shallow merge is fine here
        for k, v in part.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg

def build_model(cfg):
    if cfg["model"]["name"] == "plain":
        return PlainCNN(cfg)
    elif cfg["model"]["name"] == "ipfe":
        return IPFECNN(cfg)
    else:
        raise ValueError("Unknown model.name")

def load_model_from_cfg(cfg):
    # 1) Build model
    if cfg["model"]["name"] == "plain":
        model = PlainCNN(cfg)
    else:
        model = IPFECNN(cfg)

    # 2) Locate checkpoint using the config-based tag
    models_dir = Path(cfg["save"]["models_dir"])
    ckpt_path = find_checkpoint(cfg)

    # 3) Load weights
    model.load_from_checkpoint(str(ckpt_path))
    return model

def save_metrics_to_txt(cfg, total_params, test_metrics, loaders, train_metrics):
    """
    Save important metrics to a .txt file.
    """
    import os

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "unknown")
    out_dir = cfg.get("save", {}).get("out_dir", "results")
    model_name = cfg.get("model", {}).get("name", "unknown")
    optimizations = cfg.get("optimizations", {})
    optimizations_str = ""
    if optimizations["kernel_parallelization"] and model_name == "ipfe":
        optimizations_str += "_kernel_parallelization"
    elif optimizations["kernel_patches_parallelization"] and model_name == "ipfe":
        optimizations_str += "_kernel_patches_parallelization"
    elif optimizations["batch_parallelization"] and model_name == "ipfe":
        optimizations_str += "_batch_parallelization"
    elif optimizations["batch_kernels_parallelization"] and model_name == "ipfe":
        optimizations_str += "_batch_kernels_parallelization"
    else:
        optimizations_str += ""
    tag = _build_model_tag_from_cfg(cfg)
    txt_path = f"{out_dir}/{model_name}{optimizations_str}_{tag}.txt"
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Tag: {tag}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Optimizations: {optimizations_str}\n")
        if train_metrics is None:
            ckpt_path = find_checkpoint(cfg)
            f.write(f"Weights Loaded From: {ckpt_path}\n")
        else:
            f.write(f"Trained From Scratch\n")
            for k, v in train_metrics.items():
                f.write(f"Train {k}: {v}\n")
        f.write(f"Test set size: {len(loaders['test'].dataset)}\n")
        for k, v in test_metrics.items():
            f.write(f"Test {k}: {v}\n")
    print(f"Saved metrics to: {txt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--save_weights", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg([args.cfg])
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    loaders = make_mnist_loaders(cfg)

    

    try:
        ckpt_path = find_checkpoint(cfg)
        model = load_model_from_cfg(cfg)
        print(f"Model loaded from checkpoint: {ckpt_path}")
        total_params = sum(p.numel() for p in model.parameters())
        train_metrics = None
    except FileNotFoundError:
        model = build_model(cfg).to(device)
        print(f"Model created on device: {device}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("Training from scratch and saving weights")
        train_metrics = fit(model, loaders, cfg, device=str(device))
        print(f"Train seconds: {train_metrics['train_seconds']:.2f}")
        save_checkpoint(model, cfg)
        print(f"Saved checkpoint to: {find_checkpoint(cfg)}")


    # Evaluate (test set)
    print("Evaluating on test set...")
    test_metrics = evaluate_top1(model, loaders["test"], device=str(device), cfg=cfg)
    print(f"Test set size: {len(loaders['test'].dataset)}")
    print(f"Eval seconds: {test_metrics['eval_seconds']:.2f}")
    print(f"Test Top-1: {test_metrics['top1']:.2f}%")

    save_metrics_to_txt(cfg, total_params, test_metrics, loaders, train_metrics)


if __name__ == "__main__":
    main()