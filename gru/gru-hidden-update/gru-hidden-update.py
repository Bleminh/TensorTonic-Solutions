import numpy as np

def hidden_state_update(z_t: np.ndarray, h_prev: np.ndarray,
                        h_tilde: np.ndarray) -> np.ndarray:
    """
    Returns the float64 updated hidden state.
    """
    z_t = np.asarray(z_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    h_tilde = np.asarray(h_tilde, dtype=np.float64)

    h_t = z_t * h_prev + (1 - z_t) * h_tilde
    return h_t
    pass