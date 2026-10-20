"""Appendix figure: training curves of the snapshot-trained SDE models (colour).

(a) Duffing, process-A data, single experiment mu0; (b) pendulum.
Classes: lift-project SDE d_h = 1, 2, 4; obs-TH (= d_h 0); obs-TD.
Thin lines are individual seeds (all 5 kept), bold lines the seed median at each logged iteration.

Source: the "hist" entry of runs/<tag>/train.json, written by duffing/train.py and pendulum/train.py.
  Duffing  hist rows: ["full", it, k_max, loss] (full-rollout W2 phase), ["shoot", it, loss] (obs-TD multiple shooting)
  pendulum hist rows: [it, k_max, loss]
loss = sum_{k=1}^{k_max} sorted W2^2(model x at t_k, data minibatch at t_k), one minibatch, logged every 50 iterations.
k_max grows with the rollout-length curriculum H(it) = min(T, 4 + (T-4) it / (0.5 iters)), so the summed loss
is not comparable across horizons.  We therefore plot the loss per snapshot time, L / k_max.
Only the full-rollout objective is plotted; the Duffing obs-TD multiple-shooting phase (3000 iterations,
a different objective on one-interval rollouts) precedes its curve and is not shown.
Colour is used because five model classes with five seeds each overlap; Okabe-Ito colour-blind-safe palette.
Also prints the median wall time per class (train.json "wall_s").
usage: python figD_training_curves.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from pf_common import COLW, L4DC, OUT, panel_label, save, style

style()
# Okabe-Ito colour-blind-safe qualitative palette
COL = {"dh1": "#56B4E9", "dh2": "#0072B2", "dh4": "#009E73", "obsTH": "#E69F00", "obsTD": "#D55E00"}
LAB = {"dh1": r"lift--project, $d_h{=}1$", "dh2": r"lift--project, $d_h{=}2$", "dh4": r"lift--project, $d_h{=}4$",
       "obsTH": r"obs-TH", "obsTD": r"obs-TD"}
SYS = {
    "duffing": {"dh1": "A0_mmn1", "dh2": "A0_mmn2", "dh4": "A0_mmn4", "obsTH": "A0_mmn0", "obsTD": "A0_td"},
    "pendulum": {"dh1": "mmn_dh1", "dh2": "mmn_dh2", "dh4": "mmn_dh4", "obsTH": "mmn_dh0", "obsTD": "td"},
}


def light(c, f=0.45):
    """Mix a colour with white (opaque, so the EPS needs no transparency)."""
    r = np.array(to_rgb(c))
    return tuple(f * r + (1 - f))


def curves(system, group):
    """Return list over seeds of (iterations, k_max, loss) of the full-rollout W2 objective, and wall times."""
    out, walls = [], []
    for s in range(5):
        d = json.loads((L4DC / system / "runs" / f"{group}_s{s}" / "train.json").read_text())
        rows = [r[1:] for r in d["hist"] if r[0] == "full"] if system == "duffing" else d["hist"]
        a = np.array(rows, dtype=float)
        out.append(a)
        walls.append(d["wall_s"])
    return out, walls


fig, axs = plt.subplots(1, 2, figsize=(COLW, 2.35))
fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.255, wspace=0.28)
for ax, (system, groups), L in zip(axs, SYS.items(), "ab"):
    ends = set()
    for key in ["obsTH", "obsTD", "dh1", "dh4", "dh2"]:
        runs, walls = curves(system, groups[key])
        for a in runs:
            ax.plot(a[:, 0], a[:, 2] / a[:, 1], color=light(COL[key]), lw=0.45, ls="-", zorder=1)
        it = runs[0][:, 0]
        med = np.median(np.stack([a[:, 2] / a[:, 1] for a in runs]), 0)
        ax.plot(it, med, color=COL[key], lw=1.6, ls="-", zorder=3, label=LAB[key])
        # first logged iteration at which the curriculum horizon reaches T (k_max = 15)
        full = it[runs[0][:, 1] >= 15]
        ends.add(float(full[0]))
        print(system, key, groups[key], "median wall_s %.0f" % np.median(walls),
              "first logged full-horizon it", full[0], "final median L/kmax %.4g" % med[-1])
    for e in sorted(ends):
        ax.axvline(e, color="0.45", ls=":", lw=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathcal{L}/k_{\max}$")
    panel_label(ax, L, dx=-0.16, dy=1.03)
axs[0].set_title(r"Duffing (process A)", fontsize=8.5)
axs[1].set_title(r"pendulum", fontsize=8.5)
h, l = axs[0].get_legend_handles_labels()
order = [2, 4, 3, 0, 1]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=5, fontsize=7.2,
           frameon=False, handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.54, 0.0))
save(fig, OUT / "figD_training_curves", close=True)
print("figD written")
