import torch
from .timing import timed

@torch.no_grad()
def evaluate_top1(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with timed(device=device) as t_eval:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            print(f"x: {x}")
            print(f"y: {y}")
            pred = model(x).argmax(dim=1)
            print(f"pred: {pred}")
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    acc = 100.0 * correct / max(1, total)    
    return {"top1": acc, "eval_seconds": t_eval.elapsed}
