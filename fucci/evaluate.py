"""Evaluate all trained runs + static OT + transcriptome-only references for one dataset.
usage: python evaluate.py DS    (DS in synth, hsiao, mahd)
This is the ONLY script that opens the withheld truth. Metric definitions: the pre-specified protocol sections 4-6."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from common import (CFG, ROOT, build, load_obs, rollout, static_ot_paths, knn_phase, circ_corr, circ_mae,
                    w2_np, mmd2_rbf, circ_mean, nparams)

torch.set_num_threads(1)
ds = sys.argv[1]
SEEDS = [0, 1, 2]
obs = load_obs(ds)
X, b, K, hold = obs["X"], obs["bin"], obs["K"], obs["hold"]
d = X.shape[1]
tr = np.load(ROOT / "data" / f"{ds}_truth_WITHHELD.npz")
psi = tr["psi"]
P = float(K); t0 = 0.5; dt = CFG["dt"]
NS = int(round(2 * K / dt))                         # 120 steps -> t = 0.5 ... 12.5
tgrid = t0 + dt * np.arange(NS + 1)
wrap = lambda a: np.angle(np.exp(1j * a))

if ds == "synth":
    X0 = tr["cohort_X"][0]
else:
    rng = np.random.default_rng(999)
    i0 = np.where(b == 0)[0]
    X0 = X[rng.choice(i0, 2000)]

pool = X
sig_mmd = float(np.median(np.sqrt(((pool[:, None] - pool[None]) ** 2).sum(-1))[np.triu_indices(len(pool), 1)]))
nn_dec = NearestNeighbors(n_neighbors=10).fit(X)


def decode(P_):                                     # psi_hat for points (n, d)
    _, ii = nn_dec.kneighbors(P_)
    return np.mod(circ_mean(psi[ii], axis=1), 2 * np.pi)


if ds == "synth":
    curve = tr["curve"]
    nn_curve = NearestNeighbors(n_neighbors=1).fit(curve)
    dist_fn = lambda P_: nn_curve.kneighbors(P_)[0][:, 0]
    thr_short = np.quantile(dist_fn(X), 0.99)
else:
    nn5 = NearestNeighbors(n_neighbors=6).fit(X)
    thr_short = np.quantile(nn5.kneighbors(X)[0][:, 5], 0.99)       # leave-one-out 5-NN distance
    nn5b = NearestNeighbors(n_neighbors=5).fit(X)
    dist_fn = lambda P_: nn5b.kneighbors(P_)[0][:, 4]


def phase_metrics(est):
    o = dict(phase_mae=circ_mae(est, psi), phase_corr=circ_corr(est, psi))
    A = np.stack([np.cos(est), np.sin(est)], 1); B = np.stack([np.cos(psi), np.sin(psi)], 1)
    A = A - A.mean(0); B = B - B.mean(0)
    o["phase_corr_FL"] = float(np.linalg.det(A.T @ B) / np.sqrt(np.linalg.det(A.T @ A) * np.linalg.det(B.T @ B)))
    m = b == hold
    o["phase_mae_hold"] = circ_mae(est[m], psi[m])
    rs = []
    for k in range(K):
        mk = b == k
        c = 2 * np.pi * (k + 0.5) / K
        rs.append(spearmanr(wrap(est[mk] - c), psi[mk]).correlation)
    o["within_bin_spearman"] = float(np.nanmean(rs))
    return o


def traj_metrics(T):
    """T: (NS+1, n, d) cohort trajectory on tgrid."""
    o = {}
    if not np.isfinite(T).all():
        o["diverged_frac"] = float((~np.isfinite(T).all(-1)).any(0).mean())
        T = np.nan_to_num(T, nan=1e3, posinf=1e3, neginf=-1e3)
    else:
        o["diverged_frac"] = 0.0
    s_hold = int(round((hold + 0.5 - t0) / dt))
    o["par_w2"] = w2_np(T[s_hold], X[b == hold])
    o["par_mmd"] = mmd2_rbf(T[s_hold], X[b == hold], sig_mmd)
    fits = []
    for k in range(1, K + 1):
        kk = k % K
        if kk == hold:
            continue
        fits.append(w2_np(T[int(round(k / dt))], X[b == kk]))
    o["fit_w2"] = float(np.mean(fits))
    # E1 phase readout
    cloud = T[:int(round(P / dt))].reshape(-1, d)
    lab = np.repeat(2 * np.pi * tgrid[:int(round(P / dt))] / P, T.shape[1])
    o.update(phase_metrics(knn_phase(cloud, lab, X, k=20)))
    # temporal metrics on decoded phase
    n = T.shape[1]
    PH = decode(T.reshape(-1, d)).reshape(NS + 1, n)
    lag = 5
    Dl = wrap(PH[lag:NS - lag + 1 + lag] - PH[:NS - lag + 1])
    s_end = int(round((12.0 - t0) / dt))
    o["T1_dir_err"] = float((wrap(PH[lag:s_end + lag + 1] - PH[:s_end + 1]) < 0).mean())
    adv = wrap(np.diff(PH[:int(round(P / dt)) + 1], axis=0)).sum(0)
    o["T2_cycle"] = float(((adv >= 1.5 * np.pi) & (adv <= 2.5 * np.pi)).mean())
    o["T3_shortcut"] = float((dist_fn(T.reshape(-1, d)) > thr_short).mean())
    ss = [int(round(k / dt)) for k in range(1, 2 * K + 1)]
    o["T4_track"] = float(np.mean([np.abs(wrap(PH[s] - 2 * np.pi * tgrid[s] / P)).mean() for s in ss]))
    if ds == "synth":
        TC = tr["cohort_X"]
        errs = []
        for tau in range(1, K + 1):
            s = int(round(tau / dt))
            def cc(A, B):
                return (A - A.mean(0)).T @ (B - B.mean(0)) / len(A)
            Ct = cc(TC[0], TC[s]); Cm = cc(T[0], T[s])
            errs.append(np.linalg.norm(Cm - Ct) / np.linalg.norm(Ct))
        o["M2t_err"] = float(np.mean(errs))
    return o


res = {}
t_start = time.time()
if ds == "synth":
    res["TRUTH_cohort"] = traj_metrics(tr["cohort_X"])
    print("truth", res["TRUTH_cohort"], flush=True)

# transcriptome-only references
trm = np.array([k != hold for k in b])
cent = (2 * np.pi * (b + 0.5) / K)
ang = np.arctan2(X[:, 1], X[:, 0])
best = None
for sg in (1, -1):
    for off in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        e = circ_mae(np.mod(sg * ang + off, 2 * np.pi)[trm], cent[trm])
        if best is None or e < best[0]:
            best = (e, sg, off)
res["REF_pca_angle"] = phase_metrics(np.mod(best[1] * ang + best[2], 2 * np.pi))
tb = [k for k in range(K) if k != hold]
C = np.stack([X[b == k].mean(0) for k in tb]); ph = np.array([2 * np.pi * (k + 0.5) / K for k in tb])
bestd = np.full(len(X), np.inf); est = np.zeros(len(X))
for i in range(len(tb)):
    A, B = C[i], C[(i + 1) % len(tb)]
    pa, pb = ph[i], ph[i] + np.mod(ph[(i + 1) % len(tb)] - ph[i], 2 * np.pi)
    u = np.clip(((X - A) @ (B - A)) / ((B - A) @ (B - A)), 0, 1)
    dd = np.linalg.norm(X - (A + u[:, None] * (B - A)), axis=1)
    m = dd < bestd
    bestd[m] = dd[m]; est[m] = np.mod(pa + u[m] * (pb - pa), 2 * np.pi)
res["REF_centroid_polygon"] = phase_metrics(est)
print("refs", res["REF_pca_angle"], res["REF_centroid_polygon"], flush=True)

for s in SEEDS:
    for ent in (False, True):
        T = static_ot_paths(obs, len(X0), 2, seed=100 + s, entropic=ent)
        res[f"{'entot' if ent else 'sot'}_s{s}"] = traj_metrics(T)
        print(f"{'entot' if ent else 'sot'}_s{s}", res[f"{'entot' if ent else 'sot'}_s{s}"], flush=True)

for run in sorted((ROOT / "runs").glob(f"{ds}_*")):
    if "pilot" in run.name or not (run / "model.pt").exists():
        continue
    info = json.loads((run / "train.json").read_text())
    model = build(info["method"], d, info["dh"], P)
    model.load_state_dict(torch.load(run / "model.pt"))
    gen = torch.Generator().manual_seed(777 + info["seed"])
    with torch.no_grad():
        T = rollout(model, torch.tensor(X0), t0, t0 + 2 * P, gen).numpy()
    r = traj_metrics(T); r["nparams"] = nparams(model); r["train_wall_s"] = info["wall_s"]
    res[run.name.replace(f"{ds}_", "")] = r
    print(run.name, r, flush=True)

Path(ROOT / f"results_{ds}.json").write_text(json.dumps(res, indent=1))
print("eval wall", time.time() - t_start)
