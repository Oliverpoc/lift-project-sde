"""Appendix data figure: the two FUCCI single-cell datasets of the case study.

Rows: GSE121265 (iPSC, Hsiao et al.) and GSE146773 (U2OS, Mahdessian et al.).
(a, d) the observed state, first two principal components of the transcriptome, coloured by the continuous
       FUCCI phase psi (cyclic colormap twilight);
(b, e) the same cells coloured by the six coarse phase bins that define the snapshots (bin k = model time k + 1/2);
       the held-out bin (k = 3) is drawn as open black circles;
(c, f) the FUCCI reporter intensities (green vs red channel) coloured by psi.
The withheld truth file is read here because this is a data-illustration figure; no model uses it.
Sources (prep_real.py):
  fucci/data/{hsiao,mahd}_obs.npz             X (6 PCs), bin, K = 6, hold = 3, ids
  fucci/data/{hsiao,mahd}_truth_WITHHELD.npz  psi; hsiao: gfp, rfp (median log10 sum, adjusted);
                                               mahd: green, red (green530/red585 lognorm rescaled)
usage: python figC_fucci_data.py   -> out/figC_fucci_data.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figC_data_common import VIR
from pf_common import COLW, OUT, ROOT, panel_label, save, style

style()
D = ROOT / "fucci" / "data"
DS = [("hsiao", r"GSE121265 (iPSC)", "gfp", "rfp"), ("mahd", r"GSE146773 (U2OS)", "green", "red")]
CYC = plt.get_cmap("twilight")
npsi = Normalize(0, 2 * np.pi)

fig = plt.figure(figsize=(COLW, 4.3))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.48, wspace=0.55,
                      left=0.08, right=0.93, top=0.94, bottom=0.17)


def central(ax, x, y, q=0.5):
    """Axis limits on the central (100 - 2q)% of cells, so a few outliers do not compress the cloud."""
    for v, setter in ((x, ax.set_xlim), (y, ax.set_ylim)):
        lo, hi = np.percentile(v, [q, 100 - q]); pad = 0.06 * (hi - lo)
        setter(lo - pad, hi + pad)


for r, (ds, name, gk, rk) in enumerate(DS):
    o = np.load(D / f"{ds}_obs.npz")
    t = np.load(D / f"{ds}_truth_WITHHELD.npz", allow_pickle=True)
    assert (o["ids"] == t["ids"]).all()
    X, b, hold, K = o["X"], o["bin"], int(o["hold"]), int(o["K"])
    psi = t["psi"]
    print(f"{ds}: N={len(b)}, K={K}, hold={hold}, cells per bin {np.bincount(b).tolist()}")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(b))

    ax = fig.add_subplot(gs[r, 0])
    sc = ax.scatter(X[order, 0], X[order, 1], c=psi[order], cmap=CYC, norm=npsi, s=3, lw=0, zorder=-1)
    ax.set_xlabel(r"PC$_1$"); ax.set_ylabel(r"PC$_2$"); central(ax, X[:, 0], X[:, 1])
    ax.text(0.0, 1.03, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    panel_label(ax, "ad"[r], dx=-0.2, dy=1.03)

    ax = fig.add_subplot(gs[r, 1])
    nb = Normalize(0, K - 1)
    for k in range(K):
        s = order[b[order] == k]
        if k == hold:
            continue
        ax.scatter(X[s, 0], X[s, 1], color=VIR(nb(k)), s=3, lw=0, zorder=-1, label=rf"$k={k}$")
    s = b == hold
    ax.scatter(X[s, 0], X[s, 1], facecolor="none", edgecolor="0.0", s=4, lw=0.3, alpha=0.75,
               label=rf"$k={hold}$ (held out)")
    ax.set_xlabel(r"PC$_1$"); ax.set_ylabel(r"PC$_2$"); central(ax, X[:, 0], X[:, 1])
    if r == 1:
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=7.5, markerscale=2.5,
                   handletextpad=0.1, columnspacing=1.0, handlelength=1.0)
    panel_label(ax, "be"[r], dx=-0.2, dy=1.03)

    ax = fig.add_subplot(gs[r, 2])
    ax.scatter(t[gk][order], t[rk][order], c=psi[order], cmap=CYC, norm=npsi, s=3, lw=0, zorder=-1)
    ax.set_xlabel(r"$I_{\mathrm{G}}$"); ax.set_ylabel(r"$I_{\mathrm{R}}$"); central(ax, t[gk], t[rk])
    panel_label(ax, "cf"[r], dx=-0.2, dy=1.03)

cax = fig.add_subplot(gs[:, 3])
cb = fig.colorbar(sc, cax=cax)
cb.set_ticks([0, np.pi, 2 * np.pi]); cb.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
cb.set_label(r"$\psi$"); cb.ax.tick_params(direction="in")

save(fig, OUT / "figC_fucci_data", close=True)
print("figC_fucci_data written")
