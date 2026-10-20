"""Re-simulate the stored L4DC models to obtain their full two-time covariance C(t, t+d) on a fine grid.

The stored evaluation files keep C only on the coarse eps_C grid (duffing/runs/*/store.npz "C0",
pendulum/eval_store.npz "C_<run>": t in {0,...,30}, d in {0.5,1,1.5,2,3,4,6,8}).  The main-text heat maps and
lag curves need C on the 0.1 sampling grid, so every stored model is rolled out again with the evaluation
rollout of monotone_stat/monotone_stat.py (x0 = truth sample at t=0, N=4000, torch generator seed 999,
dt 0.1).  No training; model.pt files are only read.

Check: eps_C recomputed on the coarse in-window grid from these rollouts must equal the stored M2_in
(duffing/results.json groups.<g>.M2_in.per_seed, pendulum/gate.json summary.<cls>.M2_in.per_seed); the
differences are printed and stored.

Output: paper_figs/data/twotime_<system>.npz with
  t, d             fine grid (t in [0,15], d in [0,8], step 0.1)
  C_truth          C of process A (duffing/data/truth.npz XA0, pendulum/truth.npz Xt), same grid
  C_<group>_s<k>   C of each stored model run
  epsC_<group>_s<k> eps_C recomputed on the coarse grid (for the check)
usage: python resim_twotime.py duffing|pendulum
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
L4DC = HERE.parent
SYS = sys.argv[1]
ROOT = L4DC / SYS
sys.path.insert(0, str(ROOT))
import common as cm  # noqa: E402

from figC_data_common import two_time_cov  # noqa: E402

torch.set_num_threads(2)
T, DT = cm.CFG["T"], cm.CFG["dt_model"]
idx = lambda t: int(round(t / DT))
C_T = np.arange(0, T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
PAIRS = [(t, d) for t in C_T for d in C_D if t + d <= T + 1e-9]
TMAX, DMAX = 15.0, 8.0


def coarse(X):
    return np.array([np.mean(X[idx(t)] * X[idx(t + d)]) - X[idx(t)].mean() * X[idx(t + d)].mean() for t, d in PAIRS])


if SYS == "duffing":
    Z = np.load(ROOT / "data" / "truth.npz")
    XA = Z["XA0"]
    groups = ["A0_mmn2", "A0_mmn1", "A0_mmn0", "A0_td", "A0_sb", "A0_cfm"]
    stored = json.loads((ROOT / "results.json").read_text())["groups"]
    ref = {g: stored[g]["M2_in"]["per_seed"] for g in groups}
else:
    Z = np.load(ROOT / "truth.npz")
    XA = Z["Xt"]
    groups = ["mmn_dh2", "mmn_dh1", "mmn_dh0", "td", "tdw", "cfm"]
    S = json.loads((ROOT / "gate.json").read_text())["summary"]
    key = {"mmn_dh2": "MMN_dh2", "mmn_dh1": "MMN_dh1", "mmn_dh0": "obsTH(dh0)", "td": "obsTD",
           "tdw": "obsTD_warm", "cfm": "OT-CFM"}
    ref = {g: S[key[g]]["M2_in"]["per_seed"] for g in groups}
x0 = XA[0]
cA = coarse(XA)
t, d, C_truth = two_time_cov(XA, DT, TMAX, DMAX)
out = dict(t=t, d=d, C_truth=C_truth)


def roll(info, model):
    gen = torch.Generator().manual_seed(999)
    x = torch.tensor(x0, dtype=torch.float32)
    with torch.no_grad():
        if SYS == "duffing":
            X = model.rollout(x, 3 * T, DT, e=0, gen=gen)
        elif info["method"] == "cfm":
            X = model.rollout(x, 3 * T, DT)
        else:
            X = model.rollout(x, 3 * T, DT, gen=gen)
    return X.numpy()


t0 = time.time()
maxdiff = 0.0
for g in groups:
    for s in range(5):
        tag = f"{g}_s{s}"
        dd = ROOT / "runs" / tag
        info = json.loads((dd / "train.json").read_text())
        if SYS == "duffing":
            model = cm.build(info["method"], info["dh"], info["n_exp"])
        else:
            model = cm.build(info["method"], info["dh"], width=info["width"])
        model.load_state_dict(torch.load(dd / "model.pt")); model.eval()
        X = roll(info, model)
        e = float(np.linalg.norm(coarse(X) - cA) / np.linalg.norm(cA))
        out[f"C_{tag}"] = two_time_cov(X, DT, TMAX, DMAX)[2]
        out[f"epsC_{tag}"] = np.array(e)
        diff = abs(e - ref[g][s]); maxdiff = max(maxdiff, diff)
        print(f"{tag:14s} eps_C recomputed {e:.4f} stored {ref[g][s]:.4f} diff {diff:.1e}", flush=True)
        del model, X
out["max_abs_diff_vs_stored"] = np.array(maxdiff)
(HERE / "data").mkdir(exist_ok=True)
np.savez_compressed(HERE / "data" / f"twotime_{SYS}.npz", **out)
print(f"done {SYS}: max |eps_C - stored| = {maxdiff:.2e}, {time.time() - t0:.0f}s")
