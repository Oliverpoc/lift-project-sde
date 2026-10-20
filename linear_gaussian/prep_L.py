"""Snapshot data for the optional neural overlay on the linear-Gaussian example.
Underdamped Langevin oscillator = the duffing simulator with beta = 0 (gamma=0.2, omega^2=alpha=1, sigma=0.15,
x0~N(1.5,0.2^2), v0~N(0,0.2^2)); 16 unpaired snapshots t=0..15, 2000 fresh particles each, observe x only.
Consumed UNCHANGED by duffing/train.py with WORLD=L (load_data reads data/snap_L0.npy).
usage: python prep_L.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "duffing"))
from common import CFG, TK, simulate  # noqa: E402

LIN = dict(beta=0.0)
out = np.zeros((CFG["K"], CFG["N_train"]))
for k, t in enumerate(TK):
    if t == 0:
        out[k] = np.random.default_rng(82345 + k).normal(CFG["x0_mean"], CFG["x0_std"], CFG["N_train"])
    else:
        _, X, _ = simulate(CFG["N_train"], t, 82345 + 1000 * (k + 1), save_dt=t, params=LIN)
        out[k] = X[-1]
(HERE / "data").mkdir(exist_ok=True)
np.save(HERE / "data" / "snap_L0.npy", out)
# evaluation rollouts' reference ensemble (for marginal W2 of the neural overlay); exact covariances come from lg_exact.py
tg, XL, _ = simulate(CFG["N_eval"], CFG["T"], 777, save_dt=CFG["dt_model"], params=LIN)
np.savez_compressed(HERE / "data" / "truth_L.npz", tg=tg, XL=XL)
print("snap_L0", out.shape, np.round(out.mean(1), 3))
