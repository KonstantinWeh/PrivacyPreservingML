import torch
from torch import nn
from src.utils.timing import timed

def train_one_epoch(model, loader, criterion, optimizer, device, print_every=200):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

        if print_every and (batch_idx % print_every == 0):
            acc = 100.0 * correct / max(1, total)
            print(f"Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f} | Acc {acc:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / max(1, total)
    return {"loss": epoch_loss, "acc": epoch_acc}

def fit(model, loaders, cfg, device):
    crit = nn.CrossEntropyLoss()
    lr = cfg["train"]["lr"]
    opt_name = cfg["train"]["optimizer"].lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    epochs = cfg["train"]["epochs"]
    print_every = cfg["train"].get("print_every", 200)

    hist = []
    with timed(device=device) as t_train:
        for epoch in range(1, epochs + 1):
            m = train_one_epoch(model, loaders["train"], crit, optimizer, device, print_every)
            print(f"Epoch {epoch}/{epochs} — Loss: {m['loss']:.4f} | Acc: {m['acc']:.2f}%")
            hist.append({"epoch": epoch, **m})
            print("-" * 50)
    return {"train_seconds": t_train.elapsed, "history": hist}
