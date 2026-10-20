"""World C for the L4DC Duffing benchmark (reviewer Major 1).

World C has EXACTLY the one-time x-marginals of world A (Duffing, mu0) at every time, but a temporal
coupling that is neither comonotone (world B) nor the Duffing coupling (world A):

    x_t = Q_t( Phi( Y_t ) ),

where Q_t is the quantile function of world A's x-marginal at time t, Phi the standard normal CDF and
Y_t the first component of a STATIONARY 2-D Ornstein-Uhlenbeck process in "rank space",

    dY = ( -I/tau + Omega J ) Y dt + sqrt(2/tau) dW,   J = [[0,-1],[1,0]],   Y_0 ~ N(0, I),

so that Y_t ~ N(0,1) for every t (hence U_t = Phi(Y_t) ~ Unif(0,1) and x_t has exactly world A's
marginal), and corr(Y_t, Y_{t+d}) = exp(-d/tau) cos(Omega d).  Omega = 0 is the scalar OU (Gaussian
copula with monotone, non-oscillatory rank decay).  This is a Gaussian-copula Markov process on the
quantile ranks; tau and Omega are fixed by select_C.py (pre-declared max-min criterion, see the pre-specified protocol).

Exact transition used for simulation (no discretisation error):
    Y_{t+h} = e^{-h/tau} R(Omega h) Y_t + sqrt(1 - e^{-2h/tau}) xi,  xi ~ N(0, I).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DUFF = HERE.parent / "duffing"
sys.path.insert(0, str(DUFF))
from common import CFG, TK  # noqa: E402  (duffing CFG, snapshot times)

T, DT = CFG["T"], CFG["dt_model"]
C_T = np.arange(0, 2 * T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK_IN = (TT + DD <= T + 1e-9)


def idx(t):
    return int(round(t / DT))


def cov2(X):
    """Two-time covariance C(t, t+d) on the grid of duffing/evaluate_run.py (identical code)."""
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if t + d <= 2 * T + 1e-9 and idx(t + d) < X.shape[0]:
                a, b = X[idx(t)], X[idx(t + d)]
                with np.errstate(all="ignore"):
                    C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def relfro(C, Cref, mask=MASK_IN):
    """epsilon_C(C vs Cref) = ||C - Cref||_F / ||Cref||_F on the in-window mask (as M2_in)."""
    v = (C - Cref)[mask]
    if not np.isfinite(v).all():
        return float("inf")
    return float(np.linalg.norm(v) / np.linalg.norm(Cref[mask]))


def rank_paths(n, nsteps, h, tau, Om, seed):
    """Stationary rank-space process: returns U (nsteps+1, n) = Phi(Y_t[0]) on t = 0, h, 2h, ..."""
    rng = np.random.default_rng(seed)
    a = np.exp(-h / tau)
    c, s = np.cos(Om * h), np.sin(Om * h)
    Y = rng.standard_normal((2, n))
    out = [norm.cdf(Y[0])]
    for _ in range(nsteps):
        Y = a * np.stack([c * Y[0] - s * Y[1], s * Y[0] + c * Y[1]]) + np.sqrt(1 - a * a) * rng.standard_normal((2, n))
        out.append(norm.cdf(Y[0]))
    return np.array(out)


def world_C_paths(XA_marg, tau, Om, seed):
    """x_t = Q_t(U_t) with Q_t the empirical quantile of the world-A marginal ensemble XA_marg[t]
    (rows on the 0.1 grid). Same construction as world B in duffing/prep.py (np.quantile of XA0)."""
    nT, n = XA_marg.shape
    U = rank_paths(n, nT - 1, DT, tau, Om, seed)
    return np.stack([np.quantile(XA_marg[i], U[i]) for i in range(nT)])
