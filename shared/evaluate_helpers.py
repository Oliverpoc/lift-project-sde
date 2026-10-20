"""cov2 / relfro exactly as in evaluate.py (importable without running an evaluation)."""
import numpy as np
from common import CFG

T, dt = CFG["T"], CFG["dt_model"]
idx = lambda t: int(round(t / dt))
C_T = np.arange(0, 2 * T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK_IN = (TT + DD <= T + 1e-9)


def cov2(X):
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if t + d <= 2 * T + 1e-9 and idx(t + d) < X.shape[0]:
                a, b = X[idx(t)], X[idx(t + d)]
                C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def relfro(C, Cref):
    v = (C - Cref)[MASK_IN]
    if not np.isfinite(v).all():
        return float("inf")
    return float(np.linalg.norm(v) / np.linalg.norm(Cref[MASK_IN]))
