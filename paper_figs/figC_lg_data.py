"""Appendix data figure: the exact linear-Gaussian example (Duffing with beta = 0, only x observed).

(a) exact marginal mean m_x(t) and the band m_x +- s_t (no sampling), with five sample paths of the true system
    coloured by the rank of x_0 (paths from the stored simulation, not used for any number in the paper);
(b) the exact true two-time covariance C(t, t+d) = [e^{A d} Sigma(t)]_xx on a fine grid, diverging colour scale
    centred at zero; the solid line is C = 0, the dashed line is t + d = T, and the dots mark the eps_C grid
    (t in {0..15}, d in {0.5,...,8}, t + d <= T), on which 41.6% of the entries are negative.
Sources:
  linear_gaussian/lg_exact.py     moments(t), A, T, C_T, C_D (exact Gaussian formulas, same parameters)
  linear_gaussian/data/truth_L.npz tg, XL (sample paths of the same system; prep_L.py)
usage: python figC_lg_data.py   -> out/figC_lg_data.{png,eps}
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.linalg import expm

from figC_data_common import VIR, rank_picks
from pf_common import COLW, L4DC, OUT, panel_label, save, style

sys.path.insert(0, str(L4DC / "linear_gaussian"))
import lg_exact as lg  # noqa: E402

style()
T, DMAX = lg.T, 8.0
tt = np.linspace(0, T, 301)
mom = [lg.moments(t) for t in tt]
mx = np.array([mm[0][0] for mm in mom]); sx = np.sqrt([mm[1][0, 0] for mm in mom])

tc = np.arange(0, T + 1e-9, 0.1); dc = np.arange(0, DMAX + 1e-9, 0.1)
Sig = [lg.moments(t)[1] for t in tc]
Ed = [expm(lg.A * d) for d in dc]
C = np.array([[(E @ S)[0, 0] for E in Ed] for S in Sig])          # C[i, j] = C(tc_i, tc_i + dc_j)
Cg = np.array([[(expm(lg.A * d) @ lg.moments(t)[1])[0, 0] for d in lg.C_D] for t in lg.C_T])
TT, DD = np.meshgrid(lg.C_T, lg.C_D, indexing="ij")
mask = TT + DD <= T + 1e-9
print(f"negative fraction on the eps_C grid: {np.mean(Cg[mask] < 0):.4f} (paper: 41.6%)")

Z = np.load(L4DC / "linear_gaussian/data/truth_L.npz")
tg, XL = Z["tg"], Z["XL"]

fig = plt.figure(figsize=(COLW, 2.35))
gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.08, right=0.92, top=0.9, bottom=0.18)
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(tt, mx - sx, mx + sx, color="0.85", lw=0, zorder=-6, label=r"$m_x\pm s_t$")
ax.plot(tt, mx, color="0.0", ls="-", lw=1.4, label=r"$m_x(t)$")
picks, qs = rank_picks(XL[0])
for i, q in zip(picks, qs):
    ax.plot(tg, XL[:, i], color=VIR(Normalize(0, 1)(q)), ls="-", lw=0.8, zorder=-1)
ax.set_xlim(0, T); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x(t)$")
ax.legend(loc="upper right", fontsize=7.5, handlelength=1.6)
panel_label(ax, "a", dx=-0.16, dy=1.03)

ax = fig.add_subplot(gs[0, 1])
v = np.abs(C).max()
pm = ax.pcolormesh(tc, dc, C.T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), shading="nearest", rasterized=True)
ax.contour(tc, dc, C.T, levels=[0.0], colors="0.0", linewidths=0.5)
ax.plot([T - DMAX, T], [DMAX, 0], ls="--", color="0.0", lw=0.8)
ax.plot(TT[mask], DD[mask], "o", ms=1.4, mfc="0.0", mec="none", ls="")
ax.set_xlim(0, T); ax.set_ylim(0, DMAX)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$d$")
ax.tick_params(which="both", direction="out", top=False, right=False)
cb = fig.colorbar(pm, ax=ax, pad=0.02, fraction=0.06)
cb.set_label(r"$C(t,t{+}d)$"); cb.ax.tick_params(direction="in")
panel_label(ax, "b", dx=-0.13, dy=1.03)

save(fig, OUT / "figC_lg_data", close=True)
print("figC_lg_data written")
