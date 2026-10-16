import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 next hidden state.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    W_r = np.asarray(W_r, dtype=np.float64)
    W_z = np.asarray(W_z, dtype=np.float64)
    W_h = np.asarray(W_h, dtype=np.float64)
    b_r = np.asarray(b_r, dtype=np.float64)
    b_z = np.asarray(b_z, dtype=np.float64)
    b_h = np.asarray(b_h, dtype=np.float64)

    combined = np.concatenate([h_prev, x_t], axis = -1)
    
    r_i = combined @ W_r.T + b_r
    r_t = sigmoid(r_i)

    z_i = combined @ W_z.T + b_z
    z_t = sigmoid(z_i)

    combined_2 = np.concatenate([r_t * h_prev, x_t], axis = -1)
    h_i = combined_2 @ W_h.T + b_h
    h_tilde = np.tanh(h_i)

    h_t = z_t * h_prev + (1 - z_t) * h_tilde

    return h_t
    
    pass