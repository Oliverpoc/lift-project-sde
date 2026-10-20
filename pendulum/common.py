"""Shared pieces for the L4DC pendulum experiment (second dynamical system).
Adapted from the earlier Duffing study (Duffing).

Ground truth: stochastic damped pendulum, observe theta only (omega hidden, d_true = 1).
    dtheta = omega dt,  domega = (-gamma omega - (g/l) sin theta) dt + sigma dW
Training data: K unpaired snapshots of theta, fresh independent particles per time.
Loss (identical for every snapshot-trained SDE model): sum_k W2^2 between model
theta-samples and data theta-samples, exact 1-D W2 by sorting, pathwise gradients.

Angle wrapping: the initial distribution and noise level are chosen so that no
particle goes over the top (max |theta| checked and recorded in results.json);
theta is therefore treated as a real line coordinate, no wrapping.

Changes w.r.t. the earlier Duffing study (applied to ALL learned models, see the pre-specified protocol):
  * stability prior: bounded drift F_max*tanh(f/F_max) plus a radial contractive tail
    -kappa*relu(|z|-R)*z/|z| active only outside a ball of radius R that contains all data;
    diffusion bounded by g_max.
  * obs-TD / OT-CFM use Fourier time features; obs-TD trained 2x as long as MMN.
  * parameter counts matched to a common target (+-10%, achieved ~+-3%) via the width.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------- config ---
CFG = dict(
    gamma=0.1, w2=2.0, sigma=0.15,          # w2 = g/l
    x0_mean=1.8, x0_std=0.15, v0_mean=0.0, v0_std=0.2,
    T=15.0, K=16,                 # snapshot times t_k = 0,1,...,15 (~3 periods)
    N_train=2000,
    dt_true=0.005, dt_model=0.1,
    N_eval=4000,
    data_seed=23456, eval_seed=888, ref_traj_seed=5252,
    delta=0.4,                    # intervention theta -> theta + delta at T/2 (0.5 brought truth to |theta|=3.1399, too close to pi)
    iters=1500, td_iters=3000, batch=1000, lr=3e-3, lr_final=3e-4, clip=1.0,
    depth=2, h0_width=32,
    cfm_iters=20000,
    target_params=6000,           # every learned model matched to this (+-10%)
    # stability prior (all learned models)
    F_max=10.0, R_tail=8.0, kappa=5.0, g_max=2.0,
    # Fourier time features for time-dependent baselines
    n_fourier=10, fourier_period=30.0,
    seeds=5,
)
TK = np.arange(CFG["K"], dtype=float) * (CFG["T"] / (CFG["K"] - 1))


# ------------------------------------------------------------ simulator ---
def simulate(n, t_end, seed, save_dt=0.1, x0=None, v0=None, intervene=None, params=None):
    p = dict(CFG)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)
    x = rng.normal(p["x0_mean"], p["x0_std"], n) if x0 is None else x0.copy()
    v = rng.normal(p["v0_mean"], p["v0_std"], n) if v0 is None else v0.copy()
    dt = p["dt_true"]
    nsteps = int(round(t_end / dt))
    every = int(round(save_dt / dt))
    int_step = None if intervene is None else int(round(intervene[0] / dt))
    ts, XS, VS = [0.0], [x.copy()], [v.copy()]
    sq = p["sigma"] * math.sqrt(dt)
    for s in range(1, nsteps + 1):
        a = -p["gamma"] * v - p["w2"] * np.sin(x)
        x = x + v * dt
        v = v + a * dt + sq * rng.standard_normal(n)
        if int_step is not None and s == int_step:
            x = x + intervene[1]
        if s % every == 0:
            ts.append(s * dt); XS.append(x.copy()); VS.append(v.copy())
    return np.array(ts), np.array(XS), np.array(VS)


def make_training_data():
    out = np.zeros((CFG["K"], CFG["N_train"]))
    for k, t in enumerate(TK):
        if t == 0:
            rng = np.random.default_rng(CFG["data_seed"] + k)
            out[k] = rng.normal(CFG["x0_mean"], CFG["x0_std"], CFG["N_train"])
        else:
            _, X, _ = simulate(CFG["N_train"], t, CFG["data_seed"] + 1000 * (k + 1), save_dt=t)
            out[k] = X[-1]
    return out


def load_training_data():
    f = ROOT / "data_train.npy"
    if not f.exists():
        np.save(f, make_training_data())
    return np.load(f)


# ------------------------------------------------------------- metrics ---
def w2_1d(a, b, nq=2000):
    q = (np.arange(nq) + 0.5) / nq
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def sorted_w2sq_torch(a, b):
    return ((torch.sort(a)[0] - torch.sort(b)[0]) ** 2).mean()


# --------------------------------------------------------------- models ---
def mlp(din, dout, width, depth, final_scale=0.1):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    last = nn.Linear(d, dout)
    with torch.no_grad():
        last.weight.mul_(final_scale); last.bias.zero_()
    layers.append(last)
    return nn.Sequential(*layers)


def time_features(t, n):
    """Raw t plus Fourier features sin/cos(2 pi k t / P), k=1..n_fourier. t: tensor (n,1)."""
    k = torch.arange(1, CFG["n_fourier"] + 1, dtype=torch.float32)[None, :]
    a = 2 * math.pi * k * t / CFG["fourier_period"]
    return torch.cat([t / CFG["T"], torch.sin(a), torch.cos(a)], 1)


N_TF = 1 + 2 * CFG["n_fourier"]


def stable_drift(f, z):
    """Stability prior: bounded drift + radial contractive tail outside |z|>R."""
    f = CFG["F_max"] * torch.tanh(f / CFG["F_max"])
    r = z.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return f - CFG["kappa"] * torch.relu(r - CFG["R_tail"]) * z / r


class LatentSDE(nn.Module):
    """MMN: z=(x,h), time-homogeneous dz = f(z)dt + g(z)dW (diagonal g).
    dh=0 -> obs-TH.  time_dep=True -> obs-TD (Fourier time features)."""

    def __init__(self, dh, width, depth=2, time_dep=False):
        super().__init__()
        self.dh, self.time_dep = dh, time_dep
        d = 1 + dh
        self.net = mlp(d + (N_TF if time_dep else 0), 2 * d, width, depth)
        if dh > 0:
            self.h0net = mlp(1 + dh, dh, CFG["h0_width"], 2, final_scale=1.0)
        self.tclip = float("inf")

    def init_state(self, x0, gen=None):
        x0 = x0.reshape(-1, 1)
        if self.dh == 0:
            return x0
        eps = torch.randn(x0.shape[0], self.dh, generator=gen)
        return torch.cat([x0, self.h0net(torch.cat([x0, eps], 1))], 1)

    def net_in(self, z, t):
        if not self.time_dep:
            return z
        tt = torch.full_like(z[:, :1], min(t, self.tclip))
        return torch.cat([z, time_features(tt, z.shape[0])], 1)

    def fg(self, z, t):
        o = self.net(self.net_in(z, t))
        d = 1 + self.dh
        g = nn.functional.softplus(o[:, d:] - 2.0)
        g = CFG["g_max"] * torch.tanh(g / CFG["g_max"])
        return stable_drift(o[:, :d], z), g

    def step(self, z, t, dt, gen=None, noise=True):
        f, g = self.fg(z, t)
        z = z + f * dt
        if noise:
            z = z + g * math.sqrt(dt) * torch.randn(z.shape, generator=gen)
        return z

    def rollout(self, x0, t_end, dt, gen=None, save_every=1, intervene=None,
                noise=True, return_h=False):
        z = self.init_state(x0, gen)
        n = int(round(t_end / dt))
        int_step = None if intervene is None else int(round(intervene[0] / dt))
        xs, zs = [z[:, 0]], [z]
        for s in range(1, n + 1):
            z = self.step(z, (s - 1) * dt, dt, gen, noise)
            if int_step is not None and s == int_step:
                z = torch.cat([z[:, :1] + intervene[1], z[:, 1:]], 1)
            if s % save_every == 0:
                xs.append(z[:, 0])
                if return_h:
                    zs.append(z)
        X = torch.stack(xs)
        return (X, torch.stack(zs)) if return_h else X


class CFMField(nn.Module):
    """minibatch-OT conditional flow matching velocity v(x,t), Fourier time features,
    same stability prior (bounded field + contractive tail)."""

    def __init__(self, width, depth=2):
        super().__init__()
        self.net = mlp(1 + N_TF, 1, width, depth, final_scale=1.0)
        self.tclip = float("inf")

    def raw(self, x, tt):  # x (n,), tt (n,) tensors
        return self.net(torch.cat([x[:, None], time_features(tt[:, None], x.shape[0])], 1))[:, 0]

    def v(self, x, t):
        tt = torch.full_like(x, min(t, self.tclip))
        return stable_drift(self.raw(x, tt)[:, None], x[:, None])[:, 0]

    def rollout(self, x0, t_end, dt, intervene=None, **_):
        x = x0.clone(); xs = [x]
        n = int(round(t_end / dt))
        int_step = None if intervene is None else int(round(intervene[0] / dt))
        for s in range(1, n + 1):
            x = x + self.v(x, (s - 1) * dt) * dt
            if int_step is not None and s == int_step:
                x = x + intervene[1]
            xs.append(x)
        return torch.stack(xs)


class DelayRef(nn.Module):
    """Privileged reference: delay-embedding stochastic map trained on PAIRED trajectories."""

    def __init__(self, width, depth=2):
        super().__init__()
        self.net = mlp(2, 2, width, depth)

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


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def build(method, dh=0, width=None):
    """Construct a model; if width is None pick the width that matches target_params."""
    def make(w):
        if method in ("mmn",):
            return LatentSDE(dh, w, CFG["depth"])
        if method in ("td", "tdw"):
            return LatentSDE(0, w, CFG["depth"], time_dep=True)
        if method == "cfm":
            return CFMField(w, CFG["depth"])
        if method == "ref":
            return DelayRef(w, CFG["depth"])
        raise ValueError(method)
    if width is None:
        width = min(range(8, 200), key=lambda w: abs(nparams(make(w)) - CFG["target_params"]))
    m = make(width)
    m.width = width
    return m


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1))


if __name__ == "__main__":
    for meth, dh in [("mmn", 0), ("mmn", 1), ("mmn", 2), ("mmn", 4), ("td", 0), ("cfm", 0), ("ref", 0)]:
        m = build(meth, dh)
        print(meth, dh, "width", m.width, "params", nparams(m),
              f"{100*(nparams(m)/CFG['target_params']-1):+.1f}%")
