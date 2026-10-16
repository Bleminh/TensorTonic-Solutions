import numpy as np

def candidate_hidden(r_t: np.ndarray, h_prev: np.ndarray, x_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 candidate hidden state.
    """
    r_t = np.asarray(r_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    W_h = np.asarray(W_h, dtype=np.float64)
    b_h = np.asarray(b_h, dtype=np.float64)

    z_i = r_t * h_prev
    combined = np.concatenate([z_i, x_t], axis = -1)
    z_t = combined @ W_h.T + b_h

    h_t = np.tanh(z_t)
    return h_t
    pass