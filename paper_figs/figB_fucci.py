"""Appendix figure: real-data case study on FUCCI-labelled single-cell RNA-seq (no training, no re-evaluation).

(a) held-out-bin marginal fit W2 (pre-specified K1 metric; t = 3.5, bin 3 never seen in training), per seed,
    for both datasets (filled: GSE121265 Hsiao; open: GSE146773 Mahdessian).
(b, c) circular phase MAE against the withheld continuous FUCCI phase, per method, seeds as points,
    bar = seed mean (the pre-specified task summarises with means); filled = all cells (primary),
    open = cells of the held-out bin only; dashed line = transcriptome-only centroid-polygon reference (all cells).
(d) supervised static readout (fucci/static_readout): ridge and kNN regression X_PCA -> (cos psi, sin psi) fitted on
    80% of the training-bin cells; points = 10 random splits; filled = 20% held-out training-bin cells,
    open = held-out-bin cells; dash = same pipeline with permuted labels (chance).
Sources: fucci/summary_{hsiao,mahd}.json, fucci/gate.json, fucci/static_readout/results.json.
usage: python figB_fucci.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from pf_common import COLW, OUT, ROOT, panel_label, save, style

CYC = ROOT / "fucci"
S = {ds: json.loads((CYC / f"summary_{ds}.json").read_text()) for ds in ("hsiao", "mahd")}
G = json.loads((CYC / "gate.json").read_text())
SR = json.loads((CYC / "static_readout" / "results.json").read_text())
METH = [("mmn2", r"lift--project"), ("mmn0", r"obs-TH"), ("td", r"obs-TD"), ("cfm", r"OT-CFM"),
        ("sot", r"OT"), ("entot", r"ent.\ OT")]
DSN = {"hsiao": r"GSE121265", "mahd": r"GSE146773"}

style()
fig, axs = plt.subplots(2, 2, figsize=(COLW, 4.0))
fig.subplots_adjust(left=0.085, right=0.995, top=0.95, bottom=0.12, wspace=0.22, hspace=0.6)
X = np.arange(len(METH))


def seeds(ds, m, key):
    return np.array(S[ds][m][key]["vals"])


# (a) held-out-bin marginal fit
ax = axs[0, 0]
for j, (ds, mfc, dx) in enumerate([("hsiao", "0.0", -0.14), ("mahd", "white", 0.14)]):
    for i, (m, _) in enumerate(METH):
        v = seeds(ds, m, "par_w2")
        ax.plot(i + dx + 0 * v, v, "o", ms=3.0, mfc=mfc, mec="0.0", mew=0.7, ls="",
                label=(DSN[ds] if i == 0 else None))
        ax.plot([i + dx - 0.1, i + dx + 0.1], [v.mean()] * 2, "-", color="0.0", lw=1.3)
    print(f"(a) {ds}: K1 ratio {G[ds]['K1']['ratio']:.3f} (lifted {G[ds]['K1']['mmn']:.3f} vs best {G[ds]['K1']['best']} "
          f"{G[ds]['K1']['best_val']:.3f})")
ax.set_xticks(X); ax.set_xticklabels([l for _, l in METH], fontsize=7.5, rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", which="both", top=False, bottom=False)
ax.set_xlim(-0.5, len(METH) - 0.5); ax.set_ylim(0.0, 1.4)
ax.set_ylabel(r"$W_2(\hat p_{t_3},p_{t_3})$")
ax.legend(loc="lower left", fontsize=7.0, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, ncol=2,
          columnspacing=0.8)
panel_label(ax, "a", dx=-0.17, dy=1.02)

# (b, c) phase MAE
for ax, ds, lab in ((axs[0, 1], "hsiao", "b"), (axs[1, 0], "mahd", "c")):
    for i, (m, _) in enumerate(METH):
        for key, mfc, dx in (("phase_mae", "0.0", -0.14), ("phase_mae_hold", "white", 0.14)):
            v = seeds(ds, m, key)
            ax.plot(i + dx + 0 * v, v, "o", ms=3.0, mfc=mfc, mec="0.0", mew=0.7, ls="",
                    label=(({"phase_mae": r"all cells", "phase_mae_hold": r"held-out bin"}[key]) if i == 0 else None))
            ax.plot([i + dx - 0.1, i + dx + 0.1], [v.mean()] * 2, "-", color="0.0", lw=1.3)
        print(f"({lab}) {ds} {m:6s} MAE {seeds(ds, m, 'phase_mae').mean():.3f}  held-out bin "
              f"{seeds(ds, m, 'phase_mae_hold').mean():.3f}")
    ref = S[ds]["REF_centroid_polygon"]["phase_mae"]["mean"]
    ax.axhline(ref, ls="--", color="0.55", lw=1.0, label=r"polygon ref.")
    ax.axhline(np.pi / 2, ls=":", color="0.55", lw=1.0)
    print(f"({lab}) {ds} centroid polygon {ref:.3f}")
    ax.set_xticks(X); ax.set_xticklabels([l for _, l in METH], fontsize=7.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlim(-0.5, len(METH) - 0.5); ax.set_ylim(0.0, 1.75)
    ax.set_ylabel(r"$\langle|\hat\psi-\psi|\rangle$")
    ax.text(0.98, 0.97, DSN[ds], transform=ax.transAxes, ha="right", va="top", fontsize=8)
    if lab == "b":
        ax.legend(loc="lower left", fontsize=6.6, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2, ncol=2,
                  columnspacing=0.6)
    panel_label(ax, lab, dx=-0.17, dy=1.02)

# (d) static readout
ax = axs[1, 1]
cats = [("hsiao", "ridge"), ("hsiao", "knn"), ("mahd", "ridge"), ("mahd", "knn")]
for i, (ds, mo) in enumerate(cats):
    sm = SR[ds]["summary"]
    for key, mfc, dx in (("test_mae", "0.0", -0.14), ("hold_mae", "white", 0.14)):
        v = np.array(sm[mo][key]["vals"])
        ax.plot(i + dx + 0 * v, v, "o", ms=3.0, mfc=mfc, mec="0.0", mew=0.7, ls="",
                label=(({"test_mae": r"20\% test cells", "hold_mae": r"held-out bin"}[key]) if i == 0 else None))
        ax.plot([i + dx - 0.1, i + dx + 0.1], [v.mean()] * 2, "-", color="0.0", lw=1.3)
    nul = sm[mo + "_null"]["test_mae"]["mean"]
    ax.plot([i - 0.3, i + 0.3], [nul] * 2, "--", color="0.55", lw=1.0, label=(r"permuted $\psi$" if i == 0 else None))
    print(f"(d) {ds} {mo}: test {sm[mo]['test_mae']['mean']:.3f}+-{sm[mo]['test_mae']['sd']:.3f} "
          f"hold {sm[mo]['hold_mae']['mean']:.3f}+-{sm[mo]['hold_mae']['sd']:.3f} null(test) {nul:.3f}")
ax.axvline(1.5, color="0.8", lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels([r"ridge", r"$k$NN", r"ridge", r"$k$NN"], fontsize=7.5)
ax.tick_params(axis="x", which="both", top=False, bottom=False)
ax.text(0.5, 1.97, DSN["hsiao"], ha="center", va="top", fontsize=7.5)
ax.text(2.5, 1.97, DSN["mahd"], ha="center", va="top", fontsize=7.5)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.0, 2.05)  # legend in one column: stays left of the dataset divider
ax.set_ylabel(r"$\langle|\hat\psi-\psi|\rangle$")
ax.legend(loc="lower left", fontsize=6.6, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2, ncol=1,
          columnspacing=0.6)
panel_label(ax, "d", dx=-0.17, dy=1.02)

save(fig, OUT / "figB_fucci", close=True)
print("figB written")
