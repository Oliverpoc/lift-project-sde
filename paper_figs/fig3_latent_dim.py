"""Figure 3: autonomous latent realization -- effect of the latent dimension d_h.

(a) training-window marginal W2, (b) two-time covariance error eps_C (inside the window),
(c) extrapolation W2 at 1.5T, 2T, 3T; for d_h = 0 (obs-TH), 1, 2, 4.
Large markers joined by lines = seed median over 5 seeds; small markers = every individual seed (all kept).
No latent-recovery statistic is shown (the probe was found uninformative, see the paper).
Sources: duffing/results.json groups A0_mmn{0,1,2,4} (M0, M2_in, M1);
pendulum/gate.json summary obsTH(dh0), MMN_dh{1,2,4} (M0, M2_in, M1).
usage: python fig3_latent_dim.py
"""
import matplotlib.pyplot as plt
import numpy as np

from pf_common import COLW, OUT, duf, pen, panel_label, save, style

style()
DH = [0, 1, 2, 4]
DUFG = [f"A0_mmn{d}" for d in DH]
PENG = ["obsTH(dh0)", "MMN_dh1", "MMN_dh2", "MMN_dh4"]
SPECS = [("M0", r"$\overline{W}_2^{\,\mathrm{train}}$"), ("M2_in", r"$\epsilon_C$"),
         ("M1", r"$\overline{W}_2^{\,t>T}$")]

fig, axs = plt.subplots(1, 3, figsize=(COLW, 2.05))
fig.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.21, wspace=0.62)
for ax, (m, lab), L in zip(axs, SPECS, "abc"):
    for vals, off, sty, name in (
            ([duf(g, m) for g in DUFG], -0.08, dict(color="0.0", ls="-", marker="o", mfc="0.0"), r"Duffing"),
            ([pen(g, m) for g in PENG], 0.08, dict(color="0.5", ls="--", marker="s", mfc="white"), r"pendulum")):
        med = np.array([np.median(v) for v in vals])
        for d, v in zip(DH, vals):
            ax.plot(d + 2.5 * off + 0 * v, v, sty["marker"], ms=2.4, mfc="0.6" if name == r"Duffing" else "white",
                    mec="0.6", mew=0.5, ls="", zorder=1)
        ax.plot(np.array(DH) + off, med, lw=1.2, ms=4, label=name, zorder=3, **sty)
        print(m, name, np.round(med, 3))
    ax.set_yscale("log"); ax.set_xticks(DH); ax.set_xlabel(r"$d_h$"); ax.set_ylabel(lab); ax.yaxis.set_label_coords(-0.34, 0.5)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    panel_label(ax, L, dx=-0.40, dy=1.02)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.6, handlelength=2.0, frameon=False)
save(fig, OUT / "fig3_latent_dim", close=True)
print("fig3 written")
