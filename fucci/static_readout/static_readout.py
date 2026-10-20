"""Static readout: how much of the continuous FUCCI phase can be read from ONE instantaneous transcriptome?

This is a SEPARATE, SUPERVISED evaluation, not part of the pre-specified dynamics task.
It fits X_PCA -> (cos psi, sin psi) with the continuous FUCCI phase psi as the regression target, so it measures how
much phase information the instantaneous state carries; it is NOT trajectory inference and is not comparable in kind
to the dynamics models, which never saw psi.

LEAKAGE NOTE. The withheld truth (data/<ds>_truth_WITHHELD.npz) is loaded explicitly here, for this evaluation only.
This script does not import or modify train.py, trains no dynamics model, and writes nothing that train.py reads.
The guard in common.guard_no_truth() protects the dynamics training code; it is deliberately not installed here.

Protocol (fixed before the first run of this script):
  data       x = the same 6 transcriptome PCs as the dynamics task (data/<ds>_obs.npz); psi from the truth file.
  splits     R = 10 random splits (seeds 0..9). Training-bin cells (bin != hold) are split 80/20 at random;
             the 80% "fit" cells are the only cells used for fitting and for model selection.
  models     (a) ridge regression (with intercept, X standardised on fit cells), alpha chosen by 5-fold CV on fit cells
                 over logspace(-3, 3, 13);
             (b) kNN regression (uniform weights, Euclidean on X), k chosen by 5-fold CV on fit cells over
                 {3, 5, 10, 15, 20, 30, 50, 75, 100}.
             Targets (cos psi, sin psi); prediction psi_hat = atan2(s_hat, c_hat). CV score = circular MAE.
  test sets  (i)  the 20% held-out cells of the training bins (never used for fitting or CV);
             (ii) all cells of the held-out bin (bin = hold; no phase in its range is seen during fitting).
  metrics    circular MAE (rad) and Fisher-Lee circular correlation, same formulas as evaluate.py.
  null       the same pipeline with psi permuted among the fit cells (chance level), and the "oracle bin centre"
             reference: predicting the true bin centre gives MAE pi/12 = 0.262 rad on within-bin-uniform phases.
usage: python static_readout.py      (single process, single thread; a few seconds per dataset)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CYC = HERE.parent
sys.path.insert(0, str(CYC))
from common import load_obs, circ_mae  # noqa: E402  (load_obs opens only *_obs.npz)

R = 10
ALPHAS = np.logspace(-3, 3, 13)
KS = [3, 5, 10, 15, 20, 30, 50, 75, 100]


def fisher_lee(est, psi):
    A = np.stack([np.cos(est), np.sin(est)], 1); B = np.stack([np.cos(psi), np.sin(psi)], 1)
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.det(A.T @ B) / np.sqrt(np.linalg.det(A.T @ A) * np.linalg.det(B.T @ B)))


def make(kind, h):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=h))
    return KNeighborsRegressor(n_neighbors=h, weights="uniform")


def predict_phase(m, X):
    Y = m.predict(X)
    return np.mod(np.arctan2(Y[:, 1], Y[:, 0]), 2 * np.pi)


def cv_select(kind, grid, X, psi, seed):
    Y = np.stack([np.cos(psi), np.sin(psi)], 1)
    kf = KFold(5, shuffle=True, random_state=seed)
    scores = []
    for h in grid:
        e = []
        for a, b in kf.split(X):
            if kind == "knn" and h >= len(a):
                e.append(np.inf); continue
            m = make(kind, h).fit(X[a], Y[a])
            e.append(circ_mae(predict_phase(m, X[b]), psi[b]))
        scores.append(np.mean(e))
    return grid[int(np.argmin(scores))], float(np.min(scores))


def run(ds):
    obs = load_obs(ds)
    X, b, hold = obs["X"].astype(np.float64), obs["bin"], obs["hold"]
    tr = np.load(CYC / "data" / f"{ds}_truth_WITHHELD.npz")          # explicit, evaluation-only truth load
    psi = tr["psi"].astype(np.float64)
    itrain = np.where(b != hold)[0]; ihold = np.where(b == hold)[0]
    out = {"N": int(len(X)), "n_trainbin_cells": int(len(itrain)), "n_heldout_bin_cells": int(len(ihold)),
           "splits": []}
    for r in range(R):
        rng = np.random.default_rng(r)
        perm = rng.permutation(itrain)
        ntest = int(round(0.2 * len(itrain)))
        itest, ifit = np.sort(perm[:ntest]), np.sort(perm[ntest:])
        rec = {"seed": r, "n_fit": int(len(ifit)), "n_test": int(len(itest))}
        for shuffled in (False, True):
            pf = psi[ifit].copy()
            if shuffled:
                pf = rng.permutation(pf)
            for kind, grid in (("ridge", list(ALPHAS)), ("knn", KS)):
                h, cvs = cv_select(kind, grid, X[ifit], pf, seed=r)
                m = make(kind, h).fit(X[ifit], np.stack([np.cos(pf), np.sin(pf)], 1))
                e_t, e_h = predict_phase(m, X[itest]), predict_phase(m, X[ihold])
                key = kind + ("_null" if shuffled else "")
                rec[key] = dict(hyper=float(h), cv_mae=cvs,
                                test_mae=circ_mae(e_t, psi[itest]), test_FL=fisher_lee(e_t, psi[itest]),
                                hold_mae=circ_mae(e_h, psi[ihold]), hold_FL=fisher_lee(e_h, psi[ihold]))
        out["splits"].append(rec)
    summ = {}
    for key in ("ridge", "knn", "ridge_null", "knn_null"):
        summ[key] = {}
        for met in ("test_mae", "test_FL", "hold_mae", "hold_FL", "hyper"):
            v = np.array([s[key][met] for s in out["splits"]])
            summ[key][met] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()),
                                  vals=v.tolist())
    out["summary"] = summ
    return out


if __name__ == "__main__":
    res = {"note": "supervised static readout of continuous FUCCI phase from the 6 transcriptome PCs; "
                   "evaluation-only use of the withheld truth; see module docstring",
           "oracle_bin_centre_mae": float(np.pi / 12), "chance_mae_uniform": float(np.pi / 2)}
    for ds in ("hsiao", "mahd"):
        res[ds] = run(ds)
        s = res[ds]["summary"]
        for key in ("ridge", "knn", "ridge_null", "knn_null"):
            print(f"{ds:5s} {key:10s} test MAE {s[key]['test_mae']['mean']:.3f}+-{s[key]['test_mae']['sd']:.3f} "
                  f"FL {s[key]['test_FL']['mean']:.3f}+-{s[key]['test_FL']['sd']:.3f} | held-out bin MAE "
                  f"{s[key]['hold_mae']['mean']:.3f}+-{s[key]['hold_mae']['sd']:.3f} FL {s[key]['hold_FL']['mean']:.3f} "
                  f"| hyper median {np.median(s[key]['hyper']['vals']):g}", flush=True)
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print("written", HERE / "results.json")
