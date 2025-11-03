import argparse, yaml, torch
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/base.yaml")
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument("--weights_path")
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
    test_metrics = evaluate_top1(model, loaders["test"], device=str(device))
    print(f"Test set size: {len(loaders['test'].dataset)}")
    print(f"Eval seconds: {test_metrics['eval_seconds']:.2f}")
    print(f"Test Top-1: {test_metrics['top1']:.2f}%")



if __name__ == "__main__":
    main()
