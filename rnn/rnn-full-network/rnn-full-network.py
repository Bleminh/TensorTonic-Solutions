import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray,
                W_hy: np.ndarray, b_h: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns a dictionary containing:
    - 'outputs': shape (N, T, O), float64
    - 'final_hidden_state': shape (N, H), float64
    """
    X = np.asarray(X, dtype=np.float64)
    h_0 = np.asarray(h_0, dtype=np.float64)
    W_xh = np.asarray(W_xh, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)
    W_hy = np.asarray(W_hy, dtype=np.float64)
    b_h = np.asarray(b_h, dtype=np.float64)
    b_y = np.asarray(b_y, dtype=np.float64)

    if W_hy.ndim == 1:
        W_hy = W_hy.reshape(1, -1)
    
    b_y = b_y.reshape(-1)

    N, T, D = X.shape
    outputs_list = []
    h_prev = h_0

    for t in range(T):
        x_t = X[:, t, :]

        h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)

        y_t = h_t @ W_hy.T + b_y

        outputs_list.append(y_t)
        h_prev = h_t

    outputs = np.stack(outputs_list, axis=1)
    final_hidden_state = h_prev

    return {
        "outputs": outputs,
        "final_hidden_state": final_hidden_state
    }