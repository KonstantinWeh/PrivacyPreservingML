from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def make_mnist_loaders(cfg):
    d = cfg["data"]
    tfm = transforms.Compose([
        transforms.ToTensor(),  # (C,H,W) in [0,1]
        # Optionally normalize:
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root=d["root"], train=True, download=d["download"], transform=tfm)
    test_ds  = datasets.MNIST(root=d["root"], train=False, download=d["download"], transform=tfm)

    # Optionally limit test set to first N samples
    max_test_samples = d.get("max_test_samples", None)
    if max_test_samples is not None:
        test_ds = Subset(test_ds, indices=range(min(max_test_samples, len(test_ds))))

    bs = d["batch_size"]
    nw = d.get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return {"train": train_loader, "test": test_loader}
