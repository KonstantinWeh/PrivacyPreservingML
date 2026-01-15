import numpy as np

def prime_finder(weights):

    # Reconstruct your tensor as a NumPy array
    x = weights.detach().cpu().numpy()

    # For each 5x5 slice (axis 2,3):
    # max positive inner product: choose 255 where x > 0, 0 otherwise
    max_pos = 255 * np.maximum(x, 0).sum(axis=(2, 3))

    # max negative inner product: choose 255 where x < 0, 0 otherwise
    max_neg = 255 * np.minimum(x, 0).sum(axis=(2, 3))

    # maximum absolute inner product per slice
    max_abs = np.maximum(np.abs(max_pos), np.abs(max_neg))

    global_max_abs = max_abs.max()

    scaled_max = global_max_abs * 10000

    prime_minimum = scaled_max * 2

    print("Maximum absolute inner product over all slices:", global_max_abs)
    print("Scaled maximum absolute inner product (x10000):", scaled_max)
    print("Suggested prime minimum (scaled max * 2):", prime_minimum)

