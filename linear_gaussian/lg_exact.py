"""Exact linear-Gaussian example: same marginals, different two-time covariances, no neural optimisation.

System (partially observed underdamped Langevin oscillator; the duffing parameters with beta = 0):
    dx = v dt,   dv = (-gamma v - omega^2 x) dt + sigma dW,   (x0, v0) ~ N((1.5, 0), diag(0.2^2, 0.2^2)),
    gamma = 0.2, omega^2 = 1, sigma = 0.15; observe x only.
z = (x, v):  dz = A z dt + Lw dW,  A = [[0,1],[-omega^2,-gamma]],  Q = diag(0, sigma^2).
Exact formulas (no sampling):
  mean        m(t) = e^{At} m0
  covariance  P(t) = e^{At} P0 e^{A't} + int_0^t e^{As} Q e^{A's} ds   (Van Loan block exponential)
  marginal    p_t(x) = N(m_x(t), s_t^2),  s_t^2 = P_xx(t)
Couplings compared on the duffing eps_C grid (t in {0..15}, d in {0.5,1,1.5,2,3,4,6,8}, t+d <= 15; centred
covariances, as M2_in):
  A  true (= autonomous lifted realisation with d_h = 1):   C_A(t,t+d) = [e^{Ad} P(t)]_xx
  B  comonotone (rank preserving):                           C_B(t,t+d) = s_t s_{t+d}
  G  Gyongy / Markov mimic in x:  dx = E[v | x_t = x] dt + 0 dW (x has no own noise), with
     E[v|x] = m_v + k(t)(x - m_x), k = P_xv / P_xx.  Gauss-Markov:  C_G(t,s) = s_t^2 exp(int_t^s k).
     Since d/dt s_t^2 = 2 P_xv, k = d/dt log s_t, hence C_G = s_t s_s = C_B (proved; checked numerically).
  M(c) Gauss-Markov 1-D linear SDEs with the same marginals and diffusion c > 0:
     dx = (dm_x/dt + b(t)(x - m_x)) dt + c dW,  b = k - c^2/(2 s^2)   (mean and variance ODEs),
     C_M(t,s) = s_t s_s exp(-(c^2/2) int_t^s du / s_u^2)  in (0, s_t s_s).
  I  independent coupling: C = 0 off the diagonal  (eps = 1 on the d > 0 grid).
Every 1-D Markov diffusion in x (comparison theorem: E[x_s | x_t] nondecreasing in x_t) has C(t,s) >= 0,
so eps(.. vs A) >= || min(C_A, 0) || / || C_A ||  -- a lower bound for the whole observed-Markov class.
usage: python lg_exact.py   -> results.json (exact part), figures/fig_linear_gaussian.{png,eps}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
gamma, om2, sigma = 0.2, 1.0, 0.15
m0 = np.array([1.5, 0.0]); P0 = np.diag([0.2 ** 2, 0.2 ** 2])
A = np.array([[0.0, 1.0], [-om2, -gamma]]); Q = np.diag([0.0, sigma ** 2])
T = 15.0
C_T = np.arange(0, T + 1e-9, 1.0)
C_D = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8])


def moments(t):
    """Exact mean and covariance at time t (Van Loan)."""
    n = 2
    M = np.zeros((2 * n, 2 * n)); M[:n, :n] = -A; M[:n, n:] = Q; M[n:, n:] = A.T
    E = expm(M * t)
    Phi = E[n:, n:].T                       # e^{At}
    Qd = Phi @ E[:n, n:]                    # int_0^t e^{As} Q e^{A's} ds
    return Phi @ m0, Phi @ P0 @ Phi.T + Qd


def s2(t):
    return moments(t)[1][0, 0]


def k_gy(t):
    P = moments(t)[1]
    return P[0, 1] / P[0, 0]


TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK = TT + DD <= T + 1e-9


def grid(fun):
    C = np.full(TT.shape, np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if MASK[i, j]:
                C[i, j] = fun(t, t + d)
    return C


def C_true(t, s):
    return (expm(A * (s - t)) @ moments(t)[1])[0, 0]


def C_como(t, s):
    return np.sqrt(s2(t) * s2(s))


def C_gyongy(t, s):          # computed from its own definition (drift integral), not from C_B
    return s2(t) * np.exp(quad(k_gy, t, s, epsabs=1e-13, epsrel=1e-12, limit=200)[0])


INV = {}


def int_inv_s2(t, s):
    key = (t, s)
    if key not in INV:
        INV[key] = quad(lambda u: 1.0 / s2(u), t, s, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    return INV[key]


def C_markov(c):
    return lambda t, s: np.sqrt(s2(t) * s2(s)) * np.exp(-0.5 * c * c * int_inv_s2(t, s))


def eps(C, R):
    v = (C - R)[MASK]
    return float(np.linalg.norm(v) / np.linalg.norm(R[MASK]))


def main():
    CA, CB, CG = grid(C_true), grid(C_como), grid(C_gyongy)
    CI = np.zeros_like(CA)
    res = dict(params=dict(gamma=gamma, omega2=om2, sigma=sigma, m0=m0.tolist(), P0=P0.tolist(), T=T,
                           grid_t=C_T.tolist(), grid_d=C_D.tolist(), n_grid_points=int(MASK.sum())))
    tt = np.arange(0, T + 1e-9, 1.0)
    res["marginals"] = dict(t=tt.tolist(), mean_x=[float(moments(t)[0][0]) for t in tt],
                            sd_x=[float(np.sqrt(s2(t))) for t in tt])
    # Markov family: eps vs A as a function of c, and its minimum
    cs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    CM = {c: grid(C_markov(c)) for c in cs}
    opt = minimize_scalar(lambda c: eps(grid(C_markov(c)), CA), bounds=(0.0, 2.0), method="bounded",
                          options=dict(xatol=1e-4))
    c_star = float(opt.x); CMs = grid(C_markov(c_star))
    neg_bound = float(np.linalg.norm(np.minimum(CA, 0)[MASK]) / np.linalg.norm(CA[MASK]))
    res["epsC"] = {
        "B_comonotone_vs_A": eps(CB, CA), "A_vs_B_comonotone": eps(CA, CB),
        "G_gyongy_vs_A": eps(CG, CA), "G_gyongy_vs_B_comonotone": eps(CG, CB),
        "max_abs_rel_diff_G_minus_B": float(np.nanmax(np.abs(CG - CB)[MASK] / np.abs(CB[MASK]))),
        "I_independent_vs_A": eps(CI, CA), "lifted_true_dh1_vs_A": 0.0,
        "markov_family_vs_A": {str(c): eps(CM[c], CA) for c in cs},
        "markov_family_vs_B": {str(c): eps(CM[c], CB) for c in cs},
        "markov_best_c": c_star, "markov_best_vs_A": eps(CMs, CA),
        "lower_bound_any_xMarkov_diffusion_vs_A": neg_bound,
        "frac_negative_true_cov_entries": float(np.mean(CA[MASK] < 0)),
    }
    # same lower bound for the Duffing world A (sampled truth, duffing/data/truth.npz) -- context for obs-TD's 0.94
    try:
        sys.path.insert(0, str(HERE.parent / "process_c"))
        from processc_common import DUFF, cov2, MASK_IN
        XA0 = np.load(DUFF / "data" / "truth.npz")["XA0"]
        CDA = cov2(XA0)
        res["duffing_lower_bound_any_xMarkov_diffusion_vs_A"] = float(
            np.linalg.norm(np.minimum(CDA, 0)[MASK_IN]) / np.linalg.norm(CDA[MASK_IN]))
    except FileNotFoundError:
        res["duffing_lower_bound_any_xMarkov_diffusion_vs_A"] = None
    res["covariances"] = dict(A=CA.tolist(), B=CB.tolist(), G=CG.tolist(), M_best=CMs.tolist())
    old = json.loads((HERE / "results.json").read_text()) if (HERE / "results.json").exists() else {}
    old["exact"] = res
    (HERE / "results.json").write_text(json.dumps(old, indent=1))
    print(json.dumps(res["epsC"], indent=1))
    print("duffing bound", res["duffing_lower_bound_any_xMarkov_diffusion_vs_A"])
    np.savez(HERE / "data" / "exact_cov.npz", CA=CA, CB=CB, CG=CG, CMs=CMs, c_star=c_star, mask=MASK, C_T=C_T, C_D=C_D)


if __name__ == "__main__":
    main()
