"""Shared pieces for the L4DC Duffing week-1 experiments.

Derived from the common module of an earlier Duffing study. Changes relative to it:
  * STABILITY PRIOR (structural, identical for every learned drift):
        f(z) = B * tanh(f_theta(z)/B)  -  kappa * (|z| - R)_+ * z/|z|
    i.e. a bounded network drift plus a linear confining tail outside radius R.
    Diffusion is bounded: g = G * tanh(softplus(.)/G).  B=5, R=4, kappa=1, G=1.
    Lyapunov argument: for |z| > R + B*sqrt(d)/kappa the drift points inward, so no
    trajectory can escape to infinity (explicit Euler is stable for the tail since
    kappa*dt = 0.1 < 2).
  * Fourier time features for every time-dependent model:
        phi(t) = [t/T, sin(j*pi*t/T), cos(j*pi*t/T)]_{j=1..10}
  * Parameter counts matched to 6200 +- 10% (see PARAM table in the pre-specified protocol).
  * Second initial distribution mu1 (v0 shifted), world B and its mu1 extension.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

CFG = dict(
    gamma=0.2, alpha=1.0, beta=0.5, sigma=0.15,
    x0_mean=1.5, x0_std=0.2, v0_mean=0.0, v0_std=0.2,       # mu0 (as earlier Duffing study)
    mu1_x0_mean=1.5, mu1_x0_std=0.2, mu1_v0_mean=1.0, mu1_v0_std=0.2,  # mu1: hidden v shifted
    T=15.0, K=16, N_train=2000, dt_true=0.005, dt_model=0.1, N_eval=4000,
    data_seed=12345, eval_seed=777, ref_traj_seed=4242,
    delta=0.6,
    iters=1500, batch=1000, lr=3e-3, lr_final=3e-4, clip=1.0,
    cfm_iters=20000,
    td_shoot_iters=3000, td_shoot_batch=200,   # obs-TD multiple-shooting phase
    sb_sigma=0.25,                              # Schrodinger-bridge reference noise
    # stability prior
    B_drift=5.0, R_tail=4.0, kappa=1.0, G_diff=1.0,
    n_four=10,
    # widths chosen for parameter matching (target 6200 +- 10%)
    w_mmn=64, w_h0=32, w_obsth=76, w_1d=67, w_ref=76,
    NQ_B=40000, B_seed=31337,
)
TK = np.arange(CFG["K"], dtype=float) * (CFG["T"] / (CFG["K"] - 1))
NTF = 1 + 2 * CFG["n_four"]


def mu_params(e):
    if e == 0:
        return dict(x0_mean=CFG["x0_mean"], x0_std=CFG["x0_std"], v0_mean=CFG["v0_mean"], v0_std=CFG["v0_std"])
    return dict(x0_mean=CFG["mu1_x0_mean"], x0_std=CFG["mu1_x0_std"],
                v0_mean=CFG["mu1_v0_mean"], v0_std=CFG["mu1_v0_std"])


# ------------------------------------------------------------ simulator ---
def simulate(n, t_end, seed, save_dt=0.1, x0=None, v0=None, intervene=None, params=None):
    p = dict(CFG)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)
    x = rng.normal(p["x0_mean"], p["x0_std"], n) if x0 is None else x0.copy()
    v = rng.normal(p["v0_mean"], p["v0_std"], n) if v0 is None else v0.copy()
    dt = p["dt_true"]
    nsteps = int(round(t_end / dt)); every = int(round(save_dt / dt))
    int_step = None if intervene is None else int(round(intervene[0] / dt))
    ts, XS, VS = [0.0], [x.copy()], [v.copy()]
    sq = p["sigma"] * math.sqrt(dt)
    for s in range(1, nsteps + 1):
        a = -p["gamma"] * v - p["alpha"] * x - p["beta"] * x ** 3
        x = x + v * dt
        v = v + a * dt + sq * rng.standard_normal(n)
        if int_step is not None and s == int_step:
            x = x + intervene[1]
        if s % every == 0:
            ts.append(s * dt); XS.append(x.copy()); VS.append(v.copy())
    return np.array(ts), np.array(XS), np.array(VS)


def snapshots_A(e, seed_base):
    """K unpaired snapshots of world A under initial law mu_e (fresh particles per time)."""
    out = np.zeros((CFG["K"], CFG["N_train"]))
    pm = mu_params(e)
    for k, t in enumerate(TK):
        if t == 0:
            rng = np.random.default_rng(seed_base + k)
            out[k] = rng.normal(pm["x0_mean"], pm["x0_std"], CFG["N_train"])
        else:
            _, X, _ = simulate(CFG["N_train"], t, seed_base + 1000 * (k + 1), save_dt=t, params=pm)
            out[k] = X[-1]
    return out


def B_quantile_ensemble():
    """Large world-A ensemble under mu0 at the snapshot times. World B is DEFINED by
    the rank-preserving (comonotone) process x_t = Q_t(F_0(x_0)) built from these mu0 marginals."""
    f = ROOT / "data" / "B_ens.npy"
    if not f.exists():
        _, X, _ = simulate(CFG["NQ_B"], CFG["T"], CFG["B_seed"], save_dt=1.0)
        np.save(f, X)   # (16, NQ) on t=0..15
    return np.load(f)


def B_map(x0, Ens0, Ens_t):
    """World-B transport T_{0,t}(x0) = Q_t(F_0(x0)); F_0 = empirical CDF of the mu0 ensemble
    (mid-rank, clipped to the ensemble range), Q_t = empirical quantile of the mu0 ensemble at t."""
    s0 = np.sort(Ens0)
    u = (np.searchsorted(s0, x0, side="left") + np.searchsorted(s0, x0, side="right")) / (2.0 * len(s0))
    u = np.clip(u, 0.5 / len(s0), 1 - 0.5 / len(s0))
    return np.quantile(Ens_t, u)


def snapshots_B(e, seed_base):
    """K unpaired snapshots of world B under mu_e. B under mu_e: x0 ~ (x-law of mu_e), x_t = T_{0,t}(x0).
    Fresh particles per snapshot (unpaired)."""
    Ens = B_quantile_ensemble()
    pm = mu_params(e)
    out = np.zeros((CFG["K"], CFG["N_train"]))
    for k in range(CFG["K"]):
        rng = np.random.default_rng(seed_base + 7 * k + 1)
        x0 = rng.normal(pm["x0_mean"], pm["x0_std"], CFG["N_train"])
        out[k] = x0 if k == 0 else B_map(x0, Ens[0], Ens[k])
    return out


DATA_SEEDS = {("A", 0): 12345, ("A", 1): 22345, ("B", 0): 52345, ("B", 1): 62345}


def load_data(world, e):
    f = ROOT / "data" / f"snap_{world}{e}.npy"
    if not f.exists():
        d = snapshots_A(e, DATA_SEEDS[(world, e)]) if world == "A" else snapshots_B(e, DATA_SEEDS[(world, e)])
        np.save(f, d)
    return np.load(f)


# ------------------------------------------------------------- metrics ---
def w2_1d(a, b, nq=2000):
    q = (np.arange(nq) + 0.5) / nq
    a = np.asarray(a); b = np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def sorted_w2sq_torch(a, b):
    return ((torch.sort(a)[0] - torch.sort(b)[0]) ** 2).mean()


# --------------------------------------------------------------- models ---
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
    """Structural stability prior: bounded network drift + linear confining tail."""
    B, R, kap = CFG["B_drift"], CFG["R_tail"], CFG["kappa"]
    r = z.norm(dim=1, keepdim=True)
    return B * torch.tanh(f_raw / B) - kap * torch.relu(r - R) * z / (r + 1e-8)


def bounded_diff(g_raw):
    G = CFG["G_diff"]
    return G * torch.tanh(nn.functional.softplus(g_raw - 2.0) / G)


def tfeat(t, tclip=float("inf")):
    """t: tensor (n,1) -> Fourier time features (n, NTF)."""
    t = torch.clamp(t, max=tclip)
    T = CFG["T"]
    j = torch.arange(1, CFG["n_four"] + 1, dtype=t.dtype)[None, :]
    w = j * math.pi / T
    return torch.cat([t / T, torch.sin(w * t), torch.cos(w * t)], 1)


def onehot(e, n, n_exp):
    o = torch.zeros(n, n_exp); o[:, e] = 1.0
    return o


class LatentSDE(nn.Module):
    """MMN: z=(x,h), time-homogeneous dz = f(z)dt + g(z)dW. dh=0 -> obs-TH.
    n_exp>1: the initial latent generator q(h0|x0,e) is experiment-specific (one-hot e input);
    the dynamics f, g are SHARED and autonomous."""
    time_dep = False

    def __init__(self, dh, n_exp=1):
        super().__init__()
        self.dh, self.n_exp = dh, n_exp
        d = 1 + dh
        w = CFG["w_obsth"] if dh == 0 else CFG["w_mmn"]
        self.net = mlp(d, 2 * d, w)
        if dh > 0:
            self.h0net = mlp(1 + dh + (n_exp if n_exp > 1 else 0), dh, CFG["w_h0"], final_scale=1.0)

    def init_state(self, x0, e=0, gen=None):
        x0 = x0.reshape(-1, 1)
        if self.dh == 0:
            return x0
        eps = torch.randn(x0.shape[0], self.dh, generator=gen)
        inp = [x0, eps] + ([onehot(e, x0.shape[0], self.n_exp)] if self.n_exp > 1 else [])
        return torch.cat([x0, self.h0net(torch.cat(inp, 1))], 1)

    def fg(self, z, t=None, e=0):
        o = self.net(z); d = 1 + self.dh
        return stable_drift(o[:, :d], z), bounded_diff(o[:, d:])

    def step(self, z, t, dt, e=0, gen=None, noise=True):
        f, g = self.fg(z, t, e)
        z = z + f * dt
        if noise:
            z = z + g * math.sqrt(dt) * torch.randn(z.shape, generator=gen)
        return z

    def rollout(self, x0, t_end, dt, e=0, gen=None, save_every=1, intervene=None, noise=True, return_h=False):
        z = self.init_state(x0, e, gen)
        n = int(round(t_end / dt))
        int_step = None if intervene is None else int(round(intervene[0] / dt))
        xs, zs = [z[:, 0]], [z]
        for s in range(1, n + 1):
            z = self.step(z, None, dt, e, gen, noise)
            if int_step is not None and s == int_step:
                z = torch.cat([z[:, :1] + intervene[1], z[:, 1:]], 1)
            if s % save_every == 0:
                xs.append(z[:, 0])
                if return_h:
                    zs.append(z)
        X = torch.stack(xs)
        return (X, torch.stack(zs)) if return_h else X


class Obs1D(nn.Module):
    """Observed-space time-dependent model on x (1-D), input (x, phi(t), [one-hot e]).
    kind='td'  : learned drift + learned diffusion (Gyongy mimic), trained by W2
    kind='cfm' : deterministic velocity (OT-CFM), trained by flow matching
    kind='sb'  : drift + fixed diffusion sb_sigma (Schrodinger bridge / bridge matching)
    n_exp>1 with label=True: experiment one-hot is an input of the field."""
    time_dep = True

    def __init__(self, kind, n_exp=1, label=False):
        super().__init__()
        self.kind, self.n_exp, self.label = kind, n_exp, label
        self.dh = 0
        din = 1 + NTF + (n_exp if label else 0)
        self.net = mlp(din, 2 if kind == "td" else 1, CFG["w_1d"], final_scale=(0.1 if kind == "td" else 1.0))
        self.tclip = float("inf")

    def raw(self, x, t, e=0):
        """x (n,1), t (n,1) tensor."""
        inp = [x, tfeat(t, self.tclip)]
        if self.label:
            inp.append(onehot(e, x.shape[0], self.n_exp))
        return self.net(torch.cat(inp, 1))

    def fg(self, z, t, e=0):
        o = self.raw(z, t, e)
        f = stable_drift(o[:, :1], z)
        if self.kind == "td":
            g = bounded_diff(o[:, 1:2])
        elif self.kind == "sb":
            g = torch.full_like(f, CFG["sb_sigma"])
        else:
            g = torch.zeros_like(f)
        return f, g

    def step(self, z, t, dt, e=0, gen=None, noise=True):
        f, g = self.fg(z, t, e)
        z = z + f * dt
        if noise and self.kind != "cfm":
            z = z + g * math.sqrt(dt) * torch.randn(z.shape, generator=gen)
        return z

    def rollout(self, x0, t_end, dt, e=0, gen=None, save_every=1, intervene=None, noise=True, t_start=None):
        z = x0.reshape(-1, 1)
        n = int(round(t_end / dt))
        int_step = None if intervene is None else int(round(intervene[0] / dt))
        xs = [z[:, 0]]
        for s in range(1, n + 1):
            t = torch.full_like(z, (s - 1) * dt) if t_start is None else t_start + (s - 1) * dt
            z = self.step(z, t, dt, e, gen, noise)
            if int_step is not None and s == int_step:
                z = z + intervene[1]
            if s % save_every == 0:
                xs.append(z[:, 0])
        return torch.stack(xs)


class DelayRef(nn.Module):
    """Privileged reference (paired trajectories), as in the earlier Duffing study, width matched."""

    def __init__(self):
        super().__init__()
        self.net = mlp(2, 2, CFG["w_ref"])

    def Fs(self, x, u):
        o = self.net(torch.stack([x, u], 1))
        return o[:, 0], nn.functional.softplus(o[:, 1] - 2.0) + 1e-3

    def rollout(self, xm1, x0, t_end, dt, gen=None, intervene=None, t_start=0.0):
        u = (x0 - xm1) / dt; x = x0.clone(); xs = [x]
        n = int(round((t_end - t_start) / dt))
        int_step = None if intervene is None else int(round((intervene[0] - t_start) / dt))
        for s in range(1, n + 1):
            F, sg = self.Fs(x, u)
            u = u + F * dt + sg * math.sqrt(dt) * torch.randn(x.shape, generator=gen)
            x = x + u * dt
            if int_step is not None and s == int_step:
                x = x + intervene[1]
            xs.append(x)
        return torch.stack(xs)


def build(method, dh=0, n_exp=1):
    if method == "mmn":
        return LatentSDE(dh, n_exp)
    if method in ("td", "tdlab"):
        return Obs1D("td", n_exp, label=(n_exp > 1 and method == "tdlab"))
    if method == "tdsh":
        return Obs1D("td", n_exp, label=False)
    if method in ("cfm", "sb"):
        return Obs1D(method, n_exp, label=n_exp > 1)
    if method == "ref":
        return DelayRef()
    raise ValueError(method)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))


def run_tag(world, exps, method, dh, seed):
    m = f"mmn{dh}" if method == "mmn" else method
    return f"{world}{exps}_{m}_s{seed}"
