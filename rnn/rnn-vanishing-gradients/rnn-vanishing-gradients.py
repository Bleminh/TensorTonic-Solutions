import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers: g_k = ||W_hh||_2^k, k = 0, ..., T-1
    """
    W_hh = np.asarray(W_hh, dtype=np.float64)
    
    rho = np.linalg.norm(W_hh, ord=2)
    
    k = np.arange(T, dtype=np.float64)
    
    return np.power(rho, k)