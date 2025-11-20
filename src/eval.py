import torch
from .timing import timed

@torch.no_grad()
def evaluate_top1(model, loader, device, precrypted):
    model.eval()
    correct, total = 0, 0
    
    # Pre-encrypt all batches outside timer if needed (to exclude encryption from timing)
    t_encrypt_elapsed = 0.0
    if precrypted:
        print("Pre-encrypting all batches...")
        with timed(device=device) as t_encrypt:
            encrypted_batches = []
            for x, y in loader:
                encrypted_x = model.encrypt_data(x)
                encrypted_batches.append((encrypted_x, y))
            batches_to_process = encrypted_batches
        t_encrypt_elapsed = t_encrypt.elapsed
        print(f"Encryption completed in {t_encrypt_elapsed:.2f} seconds")
    else:
        # Convert to list only if we need to iterate multiple times
        # Otherwise just use the loader directly in the timer
        batches_to_process = None
    
    # Timer only measures inference time (encryption excluded)
    with timed(device=device) as t_eval:
        # Use pre-encrypted batches if available, otherwise use loader directly
        if batches_to_process is not None:
            batch_iter = batches_to_process
        else:
            batch_iter = loader
            
        for x, y in batch_iter:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            print(f"x: {x}")
            print(f"y: {y}")
            pred = model(x).argmax(dim=1)
            print(f"pred: {pred}")
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    
    acc = 100.0 * correct / max(1, total)
    result = {"top1": acc, "eval_seconds": t_eval.elapsed}
    if precrypted:
        result["encrypt_seconds"] = t_encrypt_elapsed
    return result
