"""Train one model.  usage: python train.py WORLD EXPS METHOD SEED [DH] [ITERS_OVERRIDE]
WORLD in {A,B}; EXPS in {0, 01} (snapshot experiments used: mu0 only, or mu0 and mu1).
METHOD in {mmn, td, tdlab, tdsh, cfm, sb, ref}.  mmn DH=0 is obs-TH.
All snapshot models see ONLY unpaired x snapshots (data/snap_<W><e>.npy). ref is privileged (paired)."""
import sys
import time

import numpy as np
import torch

from common import (CFG, TK, ROOT, build, simulate, load_data, sorted_w2sq_torch, nparams,
                    save_json, run_tag)

torch.set_num_threads(2)
world, exps, method, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
dh = int(sys.argv[5]) if len(sys.argv) > 5 else 0
iters_override = int(sys.argv[6]) if len(sys.argv) > 6 else None
E = [int(c) for c in exps]
n_exp = len(E)
tag = run_tag(world, exps, method, dh, seed)
if iters_override:
    tag = "pilot_" + tag
torch.manual_seed(seed); np.random.seed(seed)
gen = torch.Generator().manual_seed(seed)
datas = [torch.tensor(load_data(world, e), dtype=torch.float32) for e in E]
K, N = datas[0].shape
B, dt, T = CFG["batch"], CFG["dt_model"], CFG["T"]
# CHANGELOG #2: with two experiments the batch is split (B/2 per experiment) so that a two-experiment
# run costs the same per iteration as a one-experiment run.
B = B // n_exp
steps_per_snap = int(round((TK[1] - TK[0]) / dt))
model = build(method, dh, n_exp)


def cos_lr(it, n):
    c = 0.5 * (1 + np.cos(np.pi * it / n))
    return CFG["lr_final"] + (CFG["lr"] - CFG["lr_final"]) * c


t0 = time.time()
hist = []
if method in ("mmn", "td", "tdlab", "tdsh"):
    if method != "mmn":
        # ---- phase 1: multiple shooting on the marginals. For a 1-D Markov model the state IS x,
        # so restarting each interval from the data snapshot at t_k is exact marginal shooting.
        ns = CFG["td_shoot_iters"] if iters_override is None else iters_override
        Bs = CFG["td_shoot_batch"] // n_exp
        opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
        for it in range(ns):
            for g in opt.param_groups:
                g["lr"] = cos_lr(it, ns)
            loss = 0.0
            for ei, e in enumerate(E):
                d = datas[ei]
                x0 = torch.cat([d[k, torch.randperm(N, generator=gen)[:Bs]] for k in range(K - 1)])
                ts = torch.tensor(np.repeat(TK[:-1], Bs), dtype=torch.float32)[:, None]
                z = x0[:, None]
                for s in range(steps_per_snap):
                    z = model.step(z, ts + s * dt, dt, e, gen)
                z = z[:, 0].reshape(K - 1, Bs)
                for k in range(K - 1):
                    loss = loss + sorted_w2sq_torch(z[k], d[k + 1, torch.randperm(N, generator=gen)[:Bs]])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            opt.step()
            if it % 100 == 0 or it == ns - 1:
                hist.append(("shoot", it, loss.item()))
                print(tag, "shoot", it, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)
    # ---- full-rollout W2 training (identical recipe for MMN and obs-TD; curriculum 4 -> T)
    iters = CFG["iters"] if iters_override is None else iters_override
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"] if method == "mmn" else CFG["lr"] / 3)
    lr_scale = 1.0 if method == "mmn" else 1 / 3   # obs-TD fine-tunes the shooting solution
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = cos_lr(it, iters) * lr_scale
        H = min(T, 4.0 + (T - 4.0) * it / (0.5 * iters))
        kmax = int(np.floor(H / (TK[1] - TK[0]) + 1e-9))
        loss = 0.0
        for ei, e in enumerate(E):
            d = datas[ei]
            x0 = d[0, torch.randperm(N, generator=gen)[:B]]
            X = model.rollout(x0, TK[kmax], dt, e=e, gen=gen, save_every=steps_per_snap)
            for k in range(1, kmax + 1):
                loss = loss + sorted_w2sq_torch(X[k], d[k, torch.randperm(N, generator=gen)[:B]])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append(("full", it, kmax, loss.item()))
            print(tag, it, kmax, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)

elif method in ("cfm", "sb"):
    iters = CFG["cfm_iters"] if iters_override is None else iters_override
    dtk = TK[1] - TK[0]
    m = B // (K - 1) + 1
    if method == "sb":
        pairs = [np.load(ROOT / "data" / f"sb_pairs_{world}{e}.npy") for e in E]   # (K-1, P, 2)
        pairs = [torch.tensor(p, dtype=torch.float32) for p in pairs]
        sig = CFG["sb_sigma"]
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = cos_lr(it, iters)
        loss = 0.0
        for ei, e in enumerate(E):
            d = datas[ei]
            if method == "cfm":   # exact 1-D OT coupling = sorted matching
                a = torch.stack([torch.sort(d[k, torch.randperm(N, generator=gen)[:m]])[0] for k in range(K - 1)])
                b = torch.stack([torch.sort(d[k + 1, torch.randperm(N, generator=gen)[:m]])[0] for k in range(K - 1)])
            else:                 # entropic-OT (Schrodinger) coupling, pre-sampled pairs
                P = pairs[ei].shape[1]
                ii = torch.randint(0, P, (K - 1, m), generator=gen)
                pr = torch.stack([pairs[ei][k, ii[k]] for k in range(K - 1)])
                a, b = pr[..., 0], pr[..., 1]
            s = torch.rand(a.shape, generator=gen)
            tt = torch.tensor(TK[:-1], dtype=torch.float32)[:, None] + s * dtk
            if method == "cfm":
                xt = (1 - s) * a + s * b + 0.01 * torch.randn(a.shape, generator=gen)
                target = (b - a) / dtk; w = torch.ones_like(s)
            else:   # Brownian bridge; drift target (b - x_t)/((1-s) dtk), weight (1-s) tames the variance
                xt = (1 - s) * a + s * b + sig * torch.sqrt(s * (1 - s) * dtk) * torch.randn(a.shape, generator=gen)
                target = (b - xt) / ((1 - s) * dtk + 1e-3); w = (1 - s)
            xt, tt, target, w = xt.reshape(-1, 1), tt.reshape(-1, 1), target.reshape(-1, 1), w.reshape(-1, 1)
            pred, _ = model.fg(xt, tt, e)
            loss = loss + (w * (pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 500 == 0 or it == iters - 1:
            hist.append(("fm", it, loss.item()))
            print(tag, it, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)

elif method == "ref":
    iters = CFG["iters"] if iters_override is None else iters_override
    _, Xtr, _ = simulate(N, T, CFG["ref_traj_seed"] + seed, save_dt=dt)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    U = (Xtr[1:] - Xtr[:-1]) / dt
    xin, uin, unext = Xtr[1:-1].reshape(-1), U[:-1].reshape(-1), U[1:].reshape(-1)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = cos_lr(it, iters)
        idx = torch.randint(0, xin.numel(), (B * 4,), generator=gen)
        F, sg = model.Fs(xin[idx], uin[idx])
        var = (sg ** 2) * dt
        r = unext[idx] - uin[idx] - F * dt
        loss = (0.5 * r ** 2 / var + 0.5 * torch.log(var)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            hist.append(("nll", it, loss.item()))

out = ROOT / "runs" / tag
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
save_json(dict(world=world, exps=exps, method=method, seed=seed, dh=dh, n_exp=n_exp,
               nparams=nparams(model), wall_s=time.time() - t0, hist=hist), out / "train.json")
print("done", tag, nparams(model), f"{time.time()-t0:.0f}s", flush=True)
