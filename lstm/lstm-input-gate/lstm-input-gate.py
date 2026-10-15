import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    h_prev = np.asarray(h_prev, dtype = np.float64)
    x_t = np.asarray(x_t, dtype = np.float64)
    W_i = np.asarray(W_i, dtype = np.float64)
    b_i = np.asarray(b_i, dtype = np.float64)
    W_c = np.asarray(W_c, dtype = np.float64)
    b_c = np.asarray(b_c, dtype = np.float64)

    combined = np.concatenate([h_prev, x_t], axis = -1)

    z_i = combined @ W_i.T + b_i
    z_c = combined @ W_c.T + b_c

    i_gate = sigmoid(z_i)
    c_cand = np.tanh(z_c)

    return {"input_gate": i_gate, "candidate_state": c_cand}
    pass