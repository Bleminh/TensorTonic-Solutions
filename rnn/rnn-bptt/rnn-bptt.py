import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray,
                      h_prev: np.ndarray, x_t: np.ndarray,
                      W_hh: np.ndarray, W_xh: np.ndarray) -> dict:
    """
    Returns the five float64 local RNN gradients:
    dx, dh_prev, dW_xh, dW_hh, db_h
    """
    dh_next = np.asarray(dh_next, dtype=np.float64)
    h_t = np.asarray(h_t, dtype=np.float64)
    h_prev = np.asarray(h_prev, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)
    W_xh = np.asarray(W_xh, dtype=np.float64)

    da_t = dh_next * (1.0 - h_t ** 2)

    is_1d = (da_t.ndim == 1)
    if is_1d:
        da_t = da_t[None, :]
        x_t = x_t[None, :]
        h_prev = h_prev[None, :]

    dx = da_t @ W_xh
    dh_prev = da_t @ W_hh
    dW_xh = da_t.T @ x_t
    dW_hh = da_t.T @ h_prev
    db_h = np.sum(da_t, axis=0)

    if is_1d:
        dx = dx.squeeze(0)
        dh_prev = dh_prev.squeeze(0)

    return {
        "dW_hh": dW_hh,
        "dW_xh": dW_xh,
        "db_h": db_h,
        "dh_prev": dh_prev,
        "dx": dx,
    }