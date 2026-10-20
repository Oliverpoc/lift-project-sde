"""Shared code for the FUCCI single-cell case study.

Model code is adapted (copied, then generalised from 1-D to d-D) from
  ../duffing/common.py.
Kept from duffing: the structural stability prior (bounded drift + confining tail, bounded
diffusion), Fourier time features for time-dependent baselines, parameter matching.
Changed for the cyclic task: observable is d-dimensional (d=6), W2 loss is the exact minibatch
OT (POT network simplex) instead of 1-D sorting, time features are PERIODIC in the cycle period P
(this hands the time-dependent baselines the true period: baseline-favouring).

LEAKAGE RULE: training code calls guard_no_truth() first; afterwards any attempt to open a file whose
path contains 'WITHHELD' raises. Training code loads data only via load_obs().
"""
import builtins
import io
import json
import math
from pathlib import Path

import numpy as np
import ot
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

CFG = dict(
    dt=0.1, cycles=2, iters=2000, batch=256, lr=3e-3, lr_final=3e-4, clip=1.0,
    td_shoot_iters=2000, cfm_iters=12000,
    B_drift=5.0, R_tail=6.0, kappa=1.0, G_diff=1.0,
    n_four=4, w_h0=32,
    # widths chosen in param_table() for ~7000 parameters (+-10%) at d=6
    w_mmn={0: 74, 1: 64, 2: 62}, w_td=70, w_cfm=73,
    entropic_eps=0.05,
)


# ------------------------------------------------------------------ leakage guard ---
def guard_no_truth():
    _open, _ioopen, _npload = builtins.open, io.open, np.load

    def chk(f):
        if isinstance(f, (str, Path)) and "WITHHELD" in str(f):
            raise PermissionError(f"LEAKAGE GUARD: training code tried to open withheld truth {f}")

    def open2(f, *a, **k):
        chk(f); return _open(f, *a, **k)

    def ioopen2(f, *a, **k):
        chk(f); return _ioopen(f, *a, **k)

    def npload2(f, *a, **k):
        chk(f); return _npload(f, *a, **k)
    builtins.open, io.open, np.load = open2, ioopen2, npload2


def load_obs(ds):
    """Observable data only: X (N,d), bin (N,), K, hold."""
    f = DATA / f"{ds}_obs.npz"
    assert "WITHHELD" not in str(f)
    z = np.load(f)
    return dict(X=z["X"].astype(np.float32), bin=z["bin"].astype(int), K=int(z["K"]), hold=int(z["hold"]))


# ------------------------------------------------------------------ task geometry ---
def snapshot_schedule(K, hold, cycles):
    """Model time: bin k centre at t_k = k + 0.5, period P = K, repeated for `cycles` cycles.
    Rollouts start at t0 = 0.5 from bin-0 cells. Returns list of (t, k) training targets (t > t0)."""
    out = []
    for c in range(cycles):
        for k in range(K):
            t = c * K + k + 0.5
            if (c == 0 and k == 0) or k == hold:
                continue
            out.append((t, k))
    out.append((cycles * K + 0.5, 0))
    return out


# ------------------------------------------------------------------------ losses ---
def w2sq_emd(a, b):
    """Exact minibatch W2^2 between point clouds a (n,d) [grad] and b (m,d); plan from POT emd."""
    C = torch.cdist(a, b) ** 2
    with torch.no_grad():
        P = ot.emd(np.full(a.shape[0], 1.0 / a.shape[0]), np.full(b.shape[0], 1.0 / b.shape[0]),
                   C.detach().double().numpy(), numItermax=200000)
    return (torch.tensor(P, dtype=C.dtype) * C).sum()


def w2_np(a, b, nmax=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) > nmax: a = a[rng.choice(len(a), nmax, replace=False)]
    if len(b) > nmax: b = b[rng.choice(len(b), nmax, replace=False)]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    M = ot.dist(a, b)
    return float(np.sqrt(max(ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)), M, numItermax=500000), 0)))


def mmd2_rbf(a, b, sig, nmax=1500, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) > nmax: a = a[rng.choice(len(a), nmax, replace=False)]
    if len(b) > nmax: b = b[rng.choice(len(b), nmax, replace=False)]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    out = 0.0
    for s in np.array([0.25, 0.5, 1, 2, 4]) * sig:
        k = lambda u, v: np.exp(-ot.dist(u, v) / (2 * s * s))
        Kaa, Kbb, Kab = k(a, a), k(b, b), k(a, b)
        n, m = len(a), len(b)
        out += (Kaa.sum() - n) / (n * (n - 1)) + (Kbb.sum() - m) / (m * (m - 1)) - 2 * Kab.mean()
    return float(out)


# ------------------------------------------------------------------------ models ---
def mlp(din, dout, width, depth=2, final_scale=0.1):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, dout)
    with torch.no_grad():
        last.weight.mul_(final_scale); last.bias.zero_()
    layers.append(last)
    return nn.Sequential(*layers)


def stable_drift(f_raw, z):
    B, R, kap = CFG["B_drift"], CFG["R_tail"], CFG["kappa"]
    r = z.norm(dim=1, keepdim=True)
    return B * torch.tanh(f_raw / B) - kap * torch.relu(r - R) * z / (r + 1e-8)


def bounded_diff(g_raw):
    G = CFG["G_diff"]
    return G * torch.tanh(nn.functional.softplus(g_raw - 2.0) / G)


def tfeat(t, P):
    """Periodic Fourier features of model time t (n,1): [sin, cos](2 pi j t / P), j=1..n_four."""
    j = torch.arange(1, CFG["n_four"] + 1, dtype=t.dtype)[None, :]
    w = 2 * math.pi * j / P
    return torch.cat([torch.sin(w * t), torch.cos(w * t)], 1)


NTF = 2 * CFG["n_four"]


class LatentSDE(nn.Module):
    """MMN: z=(x,h) in R^{d+dh}, autonomous dz = f(z)dt + g(z) dW (diagonal g); dh=0 -> obs-TH.
    h0 = G(x0, eps) learned."""
    time_dep = False

    def __init__(self, d, dh):
        super().__init__()
        self.d, self.dh = d, dh
        D = d + dh
        self.net = mlp(D, 2 * D, CFG["w_mmn"][dh])
        if dh > 0:
            self.h0net = mlp(d + dh, dh, CFG["w_h0"], final_scale=1.0)

    def init_state(self, x0, gen=None):
        if self.dh == 0:
            return x0
        eps = torch.randn(x0.shape[0], self.dh, generator=gen)
        return torch.cat([x0, self.h0net(torch.cat([x0, eps], 1))], 1)

    def step(self, z, t, dt, gen=None, noise=True):
        o = self.net(z); D = z.shape[1]
        f, g = stable_drift(o[:, :D], z), bounded_diff(o[:, D:])
        z = z + f * dt
        if noise:
            z = z + g * math.sqrt(dt) * torch.randn(z.shape, generator=gen)
        return z


class ObsField(nn.Module):
    """Observed-space time-dependent field on x in R^d with periodic time features.
    kind='td' : SDE drift + diagonal diffusion (Gyongy mimic), trained on marginals by W2
    kind='cfm': deterministic velocity, trained by minibatch-OT conditional flow matching."""
    time_dep = True

    def __init__(self, d, kind, P):
        super().__init__()
        self.d, self.kind, self.P, self.dh = d, kind, P, 0
        w = CFG["w_td"] if kind == "td" else CFG["w_cfm"]
        self.net = mlp(d + NTF, 2 * d if kind == "td" else d, w, final_scale=(0.1 if kind == "td" else 1.0))

    def init_state(self, x0, gen=None):
        return x0

    def field(self, x, t):
        o = self.net(torch.cat([x, tfeat(t, self.P)], 1))
        f = stable_drift(o[:, :self.d], x)
        g = bounded_diff(o[:, self.d:]) if self.kind == "td" else None
        return f, g

    def step(self, z, t, dt, gen=None, noise=True):
        f, g = self.field(z, t)
        z = z + f * dt
        if noise and self.kind == "td":
            z = z + g * math.sqrt(dt) * torch.randn(z.shape, generator=gen)
        return z


def build(method, d, dh, P):
    if method == "mmn":
        return LatentSDE(d, dh)
    if method in ("td", "cfm"):
        return ObsField(d, method, P)
    raise ValueError(method)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def rollout(model, x0, t0, t_end, gen=None, noise=True, return_z=False):
    """Euler-Maruyama from t0; returns X (S+1, n, d) at every dt (and Z if return_z)."""
    dt = CFG["dt"]
    z = model.init_state(x0, gen)
    n = int(round((t_end - t0) / dt))
    xs, zs = [z[:, :model.d]], [z]
    for s in range(n):
        t = torch.full((z.shape[0], 1), t0 + s * dt)
        z = model.step(z, t, dt, gen, noise)
        xs.append(z[:, :model.d])
        if return_z:
            zs.append(z)
    X = torch.stack(xs)
    return (X, torch.stack(zs)) if return_z else X


# ------------------------------------------------------- static OT (WOT-style) cohort ---
def static_ot_paths(obs, n_paths, cycles, seed, entropic=False):
    """Chain of OT couplings between consecutive TRAINING bins (wrapping; the held-out bin is
    bridged directly). Each path starts at a bin-0 cell, follows a coupling sample to the next bin,
    with McCann (linear) interpolation in between. Returns X (S+1, n, d) on the dt grid from t0=0.5."""
    rng = np.random.default_rng(seed)
    X, b, K, hold = obs["X"], obs["bin"], obs["K"], obs["hold"]
    train_bins = [k for k in range(K) if k != hold]
    plans = {}
    for i, k in enumerate(train_bins):
        k2 = train_bins[(i + 1) % len(train_bins)]
        A, B = X[b == k], X[b == k2]
        M = ot.dist(A, B)
        a, bb = np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B))
        if entropic:
            Pm = ot.sinkhorn(a, bb, M, reg=CFG["entropic_eps"] * np.median(M), numItermax=5000, method="sinkhorn_log")
        else:
            Pm = ot.emd(a, bb, M, numItermax=500000)
        Pm = Pm / Pm.sum(1, keepdims=True)
        gap = (k2 - k) % K
        plans[k] = (np.where(b == k)[0], np.where(b == k2)[0], Pm, k2, gap)
    dt = CFG["dt"]
    idx0 = np.where(b == 0)[0]
    cur = rng.choice(len(idx0), n_paths)          # local index within bin 0
    kcur = 0
    pts = [X[idx0[cur]]]                           # knots at bin centres
    gaps = []
    total = 0
    while total < cycles * K:
        src, dst, Pm, k2, gap = plans[kcur]
        nxt = np.array([rng.choice(Pm.shape[1], p=Pm[i]) for i in cur])
        pts.append(X[dst[nxt]]); gaps.append(gap)
        cur, kcur, total = nxt, k2, total + gap
    out = [pts[0]]
    for i, g in enumerate(gaps):
        ns = int(round(g / dt))
        for s in range(1, ns + 1):
            u = s / ns
            out.append((1 - u) * pts[i] + u * pts[i + 1])
    return np.stack(out).astype(np.float32)


# ------------------------------------------------------------------- phase readout ---
def circ_mean(a, axis=None):
    return np.angle(np.exp(1j * a).mean(axis=axis))


def knn_phase(cloud_X, cloud_phase, query, k=20):
    """E1 readout: circular mean of the phase labels of the k nearest model-generated points."""
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=k).fit(cloud_X)
    _, ii = nn_.kneighbors(query)
    return np.mod(circ_mean(cloud_phase[ii], axis=1), 2 * np.pi)


def circ_corr(a, b):
    """Jammalamadaka-SenGupta circular correlation."""
    sa, sb = np.sin(a - circ_mean(a)), np.sin(b - circ_mean(b))
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def circ_mae(a, b):
    return float(np.abs(np.angle(np.exp(1j * (a - b)))).mean())


def save_json(obj, p):
    Path(p).write_text(json.dumps(obj, indent=1))
