"""Main-text data figure: the Duffing snapshots and two couplings that both reproduce them.

(a) the 16 training snapshots p_{t_k} of the observed position x (violins, kernel density estimates of the
    2000 unpaired samples at each t_k, coloured by t_k) with three paths of the true system (process A);
    the paths cross, because the hidden velocity carries particles from high to low x and back.
(b) the same snapshots with the rank-preserving paths of process B started at the same initial ranks;
    these paths never cross, yet the snapshots are identical.
Colour is used for time only (viridis), so the paths stay black and are told apart by line style.
Sources (no re-simulation):
  duffing/data/snap_A0.npy  16 x 2000 unpaired training snapshots of process A (duffing/common.py)
  duffing/data/truth.npz    tg, XA0 (process A), XB0 (process B), 4000 evaluation paths (duffing/prep.py)
usage: python fig_snapshots.py   -> out/fig_snapshots.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from figC_data_common import VIR, rank_picks
from pf_common import COLW, L4DC, OUT, panel_label, save, style

style()
T = 15.0
TK = np.arange(16, dtype=float)
snap = np.load(L4DC / "duffing/data/snap_A0.npy")
Z = np.load(L4DC / "duffing/data/truth.npz")
tg = Z["tg"]
m = tg <= T + 1e-9
assert snap.shape == (16, 2000)

xs = np.linspace(-2.2, 2.4, 400)
dens = [gaussian_kde(snap[k])(xs) for k in range(16)]
WMAX = 0.42                                           # every violin is scaled to half-width 0.42 in t
nt = Normalize(0, T)

QS = (0.1, 0.5, 0.9)
STY = [dict(ls="-", lw=1.0), dict(ls=(0, (4, 1.6)), lw=1.0), dict(ls=(0, (1, 1.2)), lw=1.3)]
PROC = [("XA0", r"true process: paths cross"), ("XB0", r"process B: ranks preserved")]

fig, axes = plt.subplots(1, 2, figsize=(COLW, 2.35), sharey=True,
                         gridspec_kw=dict(wspace=0.08, left=0.075, right=0.985, top=0.9, bottom=0.17))
for j, (ax, (key, name)) in enumerate(zip(axes, PROC)):
    for k in range(16):
        w = WMAX * dens[k] / dens[k].max()
        keep = w > 0.01
        ax.fill_betweenx(xs[keep], TK[k] - w[keep], TK[k] + w[keep], color=VIR(nt(TK[k])),
                         alpha=0.75, lw=0, zorder=-3)
        ax.plot(TK[k] - w[keep], xs[keep], color=VIR(nt(TK[k])), ls="-", lw=0.6, zorder=-2)
        ax.plot(TK[k] + w[keep], xs[keep], color=VIR(nt(TK[k])), ls="-", lw=0.6, zorder=-2)
    X = Z[key][m]
    picks, _ = rank_picks(X[0], QS)
    for i, s in zip(picks, STY):
        ax.plot(tg[m], X[:, i], color="0.0", zorder=2, **s)
        ax.plot(TK, X[np.searchsorted(tg[m], TK - 1e-9), i], "o", ms=2.4, mfc="white", mec="0.0", mew=0.7, zorder=3)
    ax.set_xlim(-0.7, T + 0.7); ax.set_ylim(xs[0], xs[-1])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel(r"$t$")
    ax.text(0.98, 0.97, name, transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    panel_label(ax, "ab"[j], dx=-0.1 if j == 0 else -0.04, dy=1.03)
axes[0].set_ylabel(r"$x$")
for ax in axes:
    ax.tick_params(which="both", top=False, right=False)

print("process A / B marginal means at t_k agree:",
      np.allclose(Z["XA0"][np.searchsorted(tg, TK - 1e-9)].mean(1), Z["XB0"][np.searchsorted(tg, TK - 1e-9)].mean(1), atol=0.03))
save(fig, OUT / "fig_snapshots", close=True)
print("written fig_snapshots")
