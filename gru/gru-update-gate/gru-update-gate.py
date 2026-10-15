import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Returns the float64 update-gate values.
    """
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    W_z = np.asarray(W_z, dtype=np.float64)
    b_z = np.asarray(b_z, dtype=np.float64)

    combined = np.concatenate([h_prev, x_t], axis = -1)

    z_t = combined @ W_z.T + b_z

    return sigmoid(z_t)
    pass