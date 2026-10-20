"""Diagnostic: is the deterministic part of one Euler-Maruyama step, x -> x + f(x) dt, non-decreasing in x
for the Duffing obs-TH models (d_h = 0)?  Proposition 2 holds in continuous time; a non-monotone step map is
how a discretized x-Markov model can produce small negative lagged rank correlations.
usage: python em_monotonicity.py  -> em_monotonicity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "duffing"))
from common import CFG, build  # noqa: E402

out = {}
x = torch.linspace(-3, 3, 6001)[:, None]
for s in range(5):
    d = HERE.parent / "duffing" / "runs" / f"A0_mmn0_s{s}"
    info = json.loads((d / "train.json").read_text())
    m = build(info["method"], info["dh"], info["n_exp"]); m.load_state_dict(torch.load(d / "model.pt")); m.eval()
    with torch.no_grad():
        f, _ = m.fg(x, None, 0)
    y = (x + CFG["dt_model"] * f)[:, 0].numpy()
    dy = np.diff(y) / np.diff(x[:, 0].numpy())
    out[f"A0_mmn0_s{s}"] = dict(min_slope=float(dy.min()), frac_grid_slope_neg=float((dy < 0).mean()),
                               max_abs_drift=float(f.abs().max()))
(HERE / "em_monotonicity.json").write_text(json.dumps(out, indent=1))
print(out)
