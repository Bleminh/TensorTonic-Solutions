import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    C_prev = np.asarray(C_prev, dtype=np.float64)
    W_f = np.asarray(W_f, dtype=np.float64)
    W_i = np.asarray(W_i, dtype=np.float64)
    W_c = np.asarray(W_c, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)
    b_f = np.asarray(b_f, dtype=np.float64)
    b_i = np.asarray(b_i, dtype=np.float64)
    b_c = np.asarray(b_c, dtype=np.float64)
    b_o = np.asarray(b_o, dtype=np.float64)

    combined = np.concatenate([h_prev, x_t], axis = -1)

    z_1 = combined @ W_f.T + b_f
    f_t = sigmoid(z_1)
    
    z_2 = combined @ W_i.T + b_i
    i_t = sigmoid(z_2)

    z_3 = combined @ W_c.T + b_c
    z_4 = np.tanh(z_3)

    c_t = f_t * C_prev + i_t * z_4

    z_5 = combined @ W_o.T + b_o
    o_t = sigmoid(z_5)

    h_t = o_t * np.tanh(c_t)

    return {"hidden_state": h_t, "cell_state": c_t}
    pass