import argparse, yaml, torch
from pathlib import Path
import os

from .seed import set_seed
from .data import make_mnist_loaders
from .models import LightweightCNN, IPFECNN
from .train import fit
from .eval import evaluate_top1
from .utils import make_run_dir, save_config, save_checkpoint

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
    if cfg["model"]["name"] == "light":
        return LightweightCNN(cfg)
    elif cfg["model"]["name"] == "ipfe":
        return IPFECNN(cfg)
    else:
        raise ValueError("Unknown model.name")

def save_metrics_to_txt(cfg, args, device, total_params, test_metrics, loaders, train_metrics=None):
    """
    Save important metrics to a .txt file, including kernel size and number of conv layers.
    """
    import os

    model_cfg = cfg.get("model", {})
    kernel_size = model_cfg.get("k1", "unknown")
    model_name = model_cfg.get("name", "unknown")
    num_convs = 4  # based on the architecture in models.py
    run_name_prefix = f"k{kernel_size}_conv{num_convs}"
    run_name = cfg.get("save", {}).get("run_name", "run")
    out_dir = cfg.get("save", {}).get("out_dir", "results")
    txt_path = f"{out_dir}/{run_name_prefix}_{run_name}_{model_name}.txt"
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(f"Run Name: {run_name_prefix}_{run_name}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Kernel Size: {kernel_size}\n")
        f.write(f"Num Convs: {num_convs}\n")
        f.write(f"Config File: {args.cfg}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        if args.weights_path:
            f.write(f"Weights Loaded From: {args.weights_path}\n")
        else:
            f.write(f"Trained From Scratch\n")
            if train_metrics is not None:
                for k, v in train_metrics.items():
                    f.write(f"Train {k}: {v}\n")
        f.write(f"Test set size: {len(loaders['test'].dataset)}\n")
        for k, v in test_metrics.items():
            f.write(f"Test {k}: {v}\n")
    print(f"Saved metrics to: {txt_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="./PrivacyPreservingML/configs/base.yaml")
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument("--weights_path", default="./PrivacyPreservingML/results/mnist/light_k1-3_conv3.pt")
    args = ap.parse_args()

    cfg = load_cfg([args.cfg])
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    loaders = make_mnist_loaders(cfg)
    model = build_model(cfg).to(device)

    # Train
    print(f"Model created on device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.weights_path:
        model.load_from_lightweight(args.weights_path)
        print(f"Weights loaded from: {args.weights_path}")
    else:
        print("No weights path provided, training from scratch")
        train_metrics = fit(model, loaders, cfg, device=str(device))
        print(f"Train seconds: {train_metrics['train_seconds']:.2f}")
        # Save artifacts (weights + cfg) if asked
        if args.save_weights:
            run_dir = make_run_dir(cfg)
            save_config(run_dir, cfg)
            save_checkpoint(run_dir, model, cfg=cfg)
            print(f"Saved checkpoint to: {run_dir}")

    # Evaluate (test set)
    print("Evaluating on test set...")
    precrypted = cfg["optimizations"]["precrypted"]
    test_metrics = evaluate_top1(model, loaders["test"], device=str(device), precrypted=precrypted)
    print(f"Test set size: {len(loaders['test'].dataset)}")
    print(f"Eval seconds: {test_metrics['eval_seconds']:.2f}")
    print(f"Test Top-1: {test_metrics['top1']:.2f}%")

    save_metrics_to_txt(cfg, args, device, total_params, test_metrics, loaders)


if __name__ == "__main__":
    main()