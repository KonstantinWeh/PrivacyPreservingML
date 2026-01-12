import numpy as np

if __name__ == "__main__":

    # Reconstruct your tensor as a NumPy array
    x = np.array([[[[-0.3437, -0.0493],
          [ 0.0494, -0.1884]]],


        [[[-0.3516, -0.4875],
          [ 0.0404,  0.1693]]],


        [[[-0.3799, -0.2633],
          [ 0.1574,  0.0654]]],


        [[[-0.4755,  0.2003],
          [ 0.3416, -0.3043]]],


        [[[ 0.1899,  0.1244],
          [ 0.0241, -0.1223]]],


        [[[ 0.4065, -0.1564],
          [ 0.3834,  0.4426]]],


        [[[-0.4730,  0.2447],
          [ 0.2261,  0.5527]]],


        [[[ 0.0543,  0.3645],
          [-0.3488, -0.2619]]]])

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

    print("Max positive per slice:", max_pos.ravel())
    print("Max negative per slice:", max_neg.ravel())
    print("Max absolute per slice:", max_abs.ravel())
    print("Maximum absolute inner product over all slices:", global_max_abs)
    print("Scaled maximum absolute inner product (x10000):", scaled_max)
    print("Suggested prime minimum (scaled max * 2):", prime_minimum)

