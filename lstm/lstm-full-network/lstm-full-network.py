import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def lstm_cell(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    C_prev: np.ndarray,
    W_f: np.ndarray,
    W_i: np.ndarray,
    W_c: np.ndarray,
    W_o: np.ndarray,
    b_f: np.ndarray,
    b_i: np.ndarray,
    b_c: np.ndarray,
    b_o: np.ndarray,
) -> dict:
  """A' cur an gnìomh aon chill LSTM."""
  combined = np.concatenate([h_prev, x_t], axis=-1)

  f_t = sigmoid(combined @ W_f.T + b_f)

  i_t = sigmoid(combined @ W_i.T + b_i)
  c_tilde = np.tanh(combined @ W_c.T + b_c)

  C_t = f_t * C_prev + i_t * c_tilde

  o_t = sigmoid(combined @ W_o.T + b_o)
  h_t = o_t * np.tanh(C_t)

  return {"hidden_state": h_t, "cell_state": C_t}


def lstm_forward(
    X: np.ndarray,
    W_f: np.ndarray,
    W_i: np.ndarray,
    W_c: np.ndarray,
    W_o: np.ndarray,
    b_f: np.ndarray,
    b_i: np.ndarray,
    b_c: np.ndarray,
    b_o: np.ndarray,
    W_y: np.ndarray,
    b_y: np.ndarray,
) -> dict:
  """A' cur an gnìomh slighe air adhart iomlan de shreath LSTM."""
  X = np.asarray(X, dtype=np.float64)
  W_f = np.asarray(W_f, dtype=np.float64)
  W_i = np.asarray(W_i, dtype=np.float64)
  W_c = np.asarray(W_c, dtype=np.float64)
  W_o = np.asarray(W_o, dtype=np.float64)
  b_f = np.asarray(b_f, dtype=np.float64)
  b_i = np.asarray(b_i, dtype=np.float64)
  b_c = np.asarray(b_c, dtype=np.float64)
  b_o = np.asarray(b_o, dtype=np.float64)
  W_y = np.asarray(W_y, dtype=np.float64)
  b_y = np.asarray(b_y, dtype=np.float64)

  N, T, D = X.shape
  H = W_f.shape[0]
  O = W_y.shape[0]

  h_t = np.zeros((N, H), dtype=np.float64)
  C_t = np.zeros((N, H), dtype=np.float64)
  outputs = np.zeros((N, T, O), dtype=np.float64)

  for t in range(T):
    x_t = X[:, t, :]

    cell_out = lstm_cell(
        x_t,
        h_t,
        C_t,
        W_f,
        W_i,
        W_c,
        W_o,
        b_f,
        b_i,
        b_c,
        b_o,
    )
    h_t = cell_out["hidden_state"]
    C_t = cell_out["cell_state"]

    y_t = h_t @ W_y.T + b_y
    outputs[:, t, :] = y_t

  return {
      "outputs": outputs,
      "final_hidden_state": h_t,
      "final_cell_state": C_t,
  }