"""Appendix data figure: the three marginal-equivalent Duffing processes A, B and C.

(a-c) sample paths of process A (true stochastic Duffing), process B (rank-preserving quantile process) and
      process C (Gaussian-copula process with oscillating ranks) over the same 5-95% marginal band of process A;
      each path is coloured by the rank of its initial value x_0, so crossing paths change colour order.
(d-f) the two-time covariance C(t, t+d) of each process on the sampling grid (step 0.1), t in [0, T], d in [0, 8],
      on one diverging colour scale centred at zero; the solid line is C = 0 and the dashed line is t + d = T
      (the paper's eps_C grid uses t + d <= T).
Sources (no re-simulation):
  duffing/data/truth.npz   tg, XA0 (process A, 4000 evaluation paths), XB0 (process B, 4000 paths); duffing/prep.py
  process_c/data/truth_C.npz  XC0 (process C, 4000 paths, tau = 20, Omega = 2); process_c/prep_C.py
usage: python figC_worlds_data.py   -> out/figC_worlds_data.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from figC_data_common import VIR, rank_picks, two_time_cov
from pf_common import COLW, L4DC, OUT, panel_label, save, style

style()
T, DT, DMAX = 15.0, 0.1, 8.0
Z = np.load(L4DC / "duffing/data/truth.npz")
ZC = np.load(L4DC / "process_c/data/truth_C.npz")
tg = Z["tg"]
W = {"A": Z["XA0"], "B": Z["XB0"], "C": ZC["XC0"]}
print("process C parameters: tau =", int(ZC["tau"]), " Omega =", float(ZC["Omega"]))
m = tg <= T + 1e-9

COV = {k: two_time_cov(X, DT, T, DMAX) for k, X in W.items()}
vmax = max(np.abs(c[2]).max() for c in COV.values())
norm = TwoSlopeNorm(0.0, -vmax, vmax)
for k, (t, d, C) in COV.items():
    print(f"process {k}: C range [{C.min():.3f}, {C.max():.3f}], negative fraction (t+d<=T) "
          f"{np.mean(C[(t[:, None] + d[None, :]) <= T + 1e-9] < 0):.3f}")

fig = plt.figure(figsize=(COLW, 4.1))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], hspace=0.5, wspace=0.32,
                      left=0.08, right=0.92, top=0.94, bottom=0.1)
qa = np.quantile(W["A"][m], [0.05, 0.95], axis=1)
rk = Normalize(0, 1)
NAMES = {"A": r"process A: Duffing", "B": r"process B: rank-preserving", "C": r"process C: Gaussian copula"}
for j, (k, X) in enumerate(W.items()):
    ax = fig.add_subplot(gs[0, j])
    ax.fill_between(tg[m], qa[0], qa[1], color="0.9", lw=0, zorder=-6)
    picks, qs = rank_picks(X[0])
    for i, q in zip(picks, qs):
        ax.plot(tg[m], X[m][:, i], color=VIR(rk(q)), ls="-", lw=1.0)
    ax.set_xlim(0, T); ax.set_ylim(-2.4, 2.6)
    ax.set_xlabel(r"$t$")
    if j == 0:
        ax.set_ylabel(r"$x(t)$")
    else:
        ax.set_yticklabels([])
    ax.set_title(NAMES[k], fontsize=8.5, pad=3)
    panel_label(ax, "abc"[j], dx=-0.06 if j else -0.22, dy=1.08)

for j, (k, (t, d, C)) in enumerate(COV.items()):
    ax = fig.add_subplot(gs[1, j])
    pm = ax.pcolormesh(t, d, C.T, cmap="RdBu_r", norm=norm, shading="nearest", rasterized=True)
    ax.contour(t, d, C.T, levels=[0.0], colors="0.0", linewidths=0.5)
    ax.plot([T - DMAX, T], [DMAX, 0], ls="--", color="0.0", lw=0.8)
    ax.set_xlim(0, T); ax.set_ylim(0, DMAX)
    ax.set_xlabel(r"$t$")
    if j == 0:
        ax.set_ylabel(r"$d$")
    else:
        ax.set_yticklabels([])
    ax.tick_params(which="both", direction="out", top=False, right=False)
    panel_label(ax, "def"[j], dx=-0.06 if j else -0.22, dy=1.04)
cax = fig.add_subplot(gs[1, 3])
cb = fig.colorbar(pm, cax=cax)
cb.set_label(r"$C(t,t{+}d)$"); cb.ax.tick_params(direction="in")

save(fig, OUT / "figC_worlds_data", close=True)
print("figC_worlds_data written")
