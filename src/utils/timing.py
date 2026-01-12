import time, contextlib, torch

class _Timer(contextlib.AbstractContextManager):
    def __init__(self, device=None):
        self.device = device
        self.elapsed = None
    def __enter__(self):
        if self.device and "cuda" in str(self.device):
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        if self.device and "cuda" in str(self.device):
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        return False

def timed(device=None):
    return _Timer(device)
