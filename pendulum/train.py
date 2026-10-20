"""Train one model.  usage: python train.py METHOD SEED [DH] [ITERS_OVERRIDE]
METHOD in {mmn, td, tdw, cfm, ref}.  mmn DH=0 is obs-TH.  td = obs-TD (Fourier time
features, 2x iterations).  tdw = obs-TD warm-started by distilling the OT-CFM field of the
same seed.  cfm = minibatch-OT CFM.  ref = privileged delay reference (paired trajectories).
All snapshot models see ONLY data_train.npy (unpaired theta snapshots)."""
import sys
import time

import numpy as np
import torch

from common import (CFG, TK, ROOT, build, simulate, load_training_data, time_features,
                    sorted_w2sq_torch, nparams, save_json)

import os
torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))
method, seed = sys.argv[1], int(sys.argv[2])
dh = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tag = f"{method}_dh{dh}_s{seed}" if method == "mmn" else f"{method}_s{seed}"
torch.manual_seed(seed); np.random.seed(seed)
gen = torch.Generator().manual_seed(seed)

data = torch.tensor(load_training_data(), dtype=torch.float32)  # (K, N)
K, N = data.shape
B, dt, T = CFG["batch"], CFG["dt_model"], CFG["T"]
iters = {"cfm": CFG["cfm_iters"], "td": CFG["td_iters"], "tdw": CFG["td_iters"]}.get(method, CFG["iters"])
if len(sys.argv) > 4:
    iters = int(sys.argv[4]); tag = "pilot_" + tag
steps_per_snap = int(round((TK[1] - TK[0]) / dt))


def lr_at(it):
    c = 0.5 * (1 + np.cos(np.pi * it / iters))
    return CFG["lr_final"] + (CFG["lr"] - CFG["lr_final"]) * c


def horizon(it):
    return min(T, 4.0 + (T - 4.0) * it / (0.5 * iters))


t0 = time.time()
hist = []
model = build(method, dh)
if method in ("mmn", "td", "tdw"):
    if method == "tdw":
        cfm = build("cfm"); cfm.load_state_dict(torch.load(ROOT / "runs" / f"cfm_s{seed}" / "model.pt"))
        dopt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
        for it in range(2000):
            k = torch.randint(0, K, (B,), generator=gen)
            xs = data[k, torch.randint(0, N, (B,), generator=gen)] + 0.1 * torch.randn(B, generator=gen)
            ts = torch.tensor((TK[k.numpy()] + np.random.rand(B) - 0.5).clip(0, T), dtype=torch.float32)
            with torch.no_grad():
                vt = cfm.raw(xs, ts)
            o = model.net(torch.cat([xs[:, None], time_features(ts[:, None], B)], 1))
            l = ((o[:, 0] - vt) ** 2).mean()
            dopt.zero_grad(); l.backward(); dopt.step()
        print(tag, "distill loss", l.item(), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        H = horizon(it)
        kmax = int(np.floor(H / (TK[1] - TK[0]) + 1e-9))
        x0 = data[0, torch.randperm(N, generator=gen)[:B]]
        X = model.rollout(x0, TK[kmax], dt, gen=gen, save_every=steps_per_snap)
        loss = 0.0
        for k in range(1, kmax + 1):
            tgt = data[k, torch.randperm(N, generator=gen)[:B]]
            loss = loss + sorted_w2sq_torch(X[k], tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append((it, kmax, loss.item()))
            print(tag, it, kmax, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)

elif method == "cfm":
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    dtk = TK[1] - TK[0]
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        loss = 0.0
        for k in range(K - 1):
            a = torch.sort(data[k, torch.randperm(N, generator=gen)[:B // (K - 1) + 1]])[0]
            b = torch.sort(data[k + 1, torch.randperm(N, generator=gen)[:B // (K - 1) + 1]])[0]
            s = torch.rand(a.shape, generator=gen)
            xt = (1 - s) * a + s * b + 0.01 * torch.randn(a.shape, generator=gen)
            tt = (TK[k] + s * dtk).float()
            pred = model.raw(xt, tt)
            loss = loss + ((pred - (b - a) / dtk) ** 2).mean()
        loss = loss / (K - 1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            hist.append((it, K - 1, loss.item()))
            print(tag, it, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)

elif method == "ref":
    _, Xtr, _ = simulate(N, T, CFG["ref_traj_seed"] + seed, save_dt=dt)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    U = (Xtr[1:] - Xtr[:-1]) / dt
    xin, uin, unext = Xtr[1:-1].reshape(-1), U[:-1].reshape(-1), U[1:].reshape(-1)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        idx = torch.randint(0, xin.numel(), (B * 4,), generator=gen)
        F, sg = model.Fs(xin[idx], uin[idx])
        var = (sg ** 2) * dt
        r = unext[idx] - uin[idx] - F * dt
        loss = (0.5 * r ** 2 / var + 0.5 * torch.log(var)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append((it, 0, loss.item()))
            print(tag, it, f"{loss.item():.4f}", f"{time.time()-t0:.0f}s", flush=True)

out = ROOT / ("pilots" if tag.startswith("pilot_") else "runs") / tag
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
save_json(dict(method=method, seed=seed, dh=dh, width=model.width, nparams=nparams(model), iters=iters,
               wall_s=time.time() - t0, hist=hist, cfg=CFG), out / "train.json")
print("done", tag, nparams(model), f"{time.time()-t0:.0f}s")
