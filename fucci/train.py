"""Train one model on coarse-bin marginals ONLY.
usage: python train.py DS METHOD SEED [DH] [ITERS_OVERRIDE]
DS in {synth, hsiao, mahd}; METHOD in {mmn, td, cfm}; mmn DH in {0 (=obs-TH), 1, 2}.
Opens only data/DS_obs.npz (X + coarse bin labels); the withheld truth is blocked by guard_no_truth()."""
import sys
import time

import numpy as np
import torch

import common
from common import CFG, build, load_obs, snapshot_schedule, w2sq_emd, rollout, nparams, save_json, ROOT

common.guard_no_truth()
torch.set_num_threads(1)
ds, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
dh = int(sys.argv[4]) if len(sys.argv) > 4 else 0
iters_override = int(sys.argv[5]) if len(sys.argv) > 5 else None
tag = f"{ds}_{'mmn%d' % dh if method == 'mmn' else method}_s{seed}" + ("_pilot" if iters_override else "")
torch.manual_seed(seed); np.random.seed(seed)
gen = torch.Generator().manual_seed(seed)

obs = load_obs(ds)
K, hold = obs["K"], obs["hold"]
X = torch.tensor(obs["X"]); b = obs["bin"]; d = X.shape[1]
bins = {k: X[torch.tensor(np.where(b == k)[0])] for k in range(K) if k != hold}   # held-out bin never loaded
P = float(K); t0 = 0.5; dt = CFG["dt"]
sched = snapshot_schedule(K, hold, CFG["cycles"])
train_bins = [k for k in range(K) if k != hold]
model = build(method, d, dh, P)
Bt = CFG["batch"]


def sample(k, n):
    xb = bins[k]
    idx = torch.randperm(len(xb), generator=gen)[:min(n, len(xb))]
    return xb[idx]


def cos_lr(it, n, scale=1.0):
    c = 0.5 * (1 + np.cos(np.pi * it / n))
    return scale * (CFG["lr_final"] + (CFG["lr"] - CFG["lr_final"]) * c)


t_start = time.time(); hist = []
if method in ("mmn", "td"):
    if method == "td":
        # phase 1: multiple shooting on consecutive training-bin intervals (exact for a Markov model on x)
        ns = CFG["td_shoot_iters"] if iters_override is None else iters_override
        opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
        for it in range(ns):
            for g in opt.param_groups: g["lr"] = cos_lr(it, ns)
            loss = 0.0
            for i, k in enumerate(train_bins):
                k2 = train_bins[(i + 1) % len(train_bins)]
                gap = (k2 - k) % K
                z = sample(k, 128); tk = k + 0.5
                for s in range(int(round(gap / dt))):
                    z = model.step(z, torch.full((z.shape[0], 1), tk + s * dt), dt, gen)
                loss = loss + w2sq_emd(z, sample(k2, 128))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); opt.step()
            if it % 100 == 0 or it == ns - 1:
                hist.append(("shoot", it, loss.item()))
                print(tag, "shoot", it, f"{loss.item():.4f}", f"{time.time()-t_start:.0f}s", flush=True)
    iters = CFG["iters"] if iters_override is None else iters_override
    scale = 1.0 if method == "mmn" else 1 / 3
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"] * scale)
    Tmax = sched[-1][0]
    for it in range(iters):
        for g in opt.param_groups: g["lr"] = cos_lr(it, iters, scale)
        H = min(Tmax, t0 + 2.0 + (Tmax - t0 - 2.0) * it / (0.5 * iters))
        tg = [(t, k) for (t, k) in sched if t <= H + 1e-6]
        Xr = rollout(model, sample(0, Bt), t0, tg[-1][0], gen)
        loss = 0.0
        for (t, k) in tg:
            loss = loss + w2sq_emd(Xr[int(round((t - t0) / dt))], sample(k, Bt))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); opt.step()
        if it % 50 == 0 or it == iters - 1:
            hist.append(("full", it, len(tg), loss.item()))
            print(tag, it, len(tg), f"{loss.item():.4f}", f"{time.time()-t_start:.0f}s", flush=True)

elif method == "cfm":
    import ot
    iters = CFG["cfm_iters"] if iters_override is None else iters_override
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    m = 128
    for it in range(iters):
        for g in opt.param_groups: g["lr"] = cos_lr(it, iters)
        xs, ts, tg = [], [], []
        for i, k in enumerate(train_bins):
            k2 = train_bins[(i + 1) % len(train_bins)]
            gap = (k2 - k) % K
            a, bb = sample(k, m), sample(k2, m)
            M = torch.cdist(a, bb).pow(2).double().numpy()
            Pm = ot.emd(np.full(len(a), 1 / len(a)), np.full(len(bb), 1 / len(bb)), M)
            j = torch.tensor([np.random.choice(len(bb), p=r / r.sum()) for r in Pm])
            bb = bb[j]
            s = torch.rand(len(a), 1, generator=gen)
            xs.append((1 - s) * a + s * bb + 0.05 * torch.randn(a.shape, generator=gen))
            ts.append(k + 0.5 + s * gap); tg.append((bb - a) / gap)
        xt, tt, target = torch.cat(xs), torch.cat(ts), torch.cat(tg)
        pred, _ = model.field(xt, tt)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"]); opt.step()
        if it % 500 == 0 or it == iters - 1:
            hist.append(("fm", it, loss.item()))
            print(tag, it, f"{loss.item():.4f}", f"{time.time()-t_start:.0f}s", flush=True)

out = ROOT / "runs" / tag
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
save_json(dict(ds=ds, method=method, seed=seed, dh=dh, nparams=nparams(model), wall_s=time.time() - t_start,
               hist=hist), out / "train.json")
print("done", tag, nparams(model), f"{time.time()-t_start:.0f}s", flush=True)
