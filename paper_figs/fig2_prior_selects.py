"""Figure 3 (label fig:select): each model class returns its own coupling, on every marginal-equivalent data set.

(a) Duffing: eps_C relative to process A versus relative to process B; filled = trained on process-A data,
    open = trained on process-B data; large marker = seed median, small markers = seeds (A data).
(b) Pendulum: same axes, models trained on process-A data.
(c) Duffing process-C test: the four classes trained on process-A (filled), process-B (open) and process-C data
    (light fill; small markers = C-data seeds); stars = processes A, B (black) and C (open).
The training-window marginal fit of the same runs is Fig. 2(e) (fig2_advantage.py).
Colour: author-approved exception to greyscale-first; class palette pf_common.PAL, identical in every main figure;
markers also differ per class.
Sources: duffing/results.json (groups A0_*, B0_*: M2_in, M2_in_vsB per seed), duffing/data/truth.npz (process
A/B reference distance); pendulum/gate.json summary (M2_in, M2_in_vs_worldB), pendulum/eval_store.npz (C_true, C_B);
process_c/results.json (groups {A0,B0,C0}_{mmn2,td,cfm,sb}: dA, dB per seed; world_C.verification.epsC).
usage: python fig2_prior_selects.py   -> out/fig2_prior_selects.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from pf_common import (C_D, C_T, COLW, DUF_T, DUF_Z, LAB, MK, OUT, PAL, PEN_ST, WC_R, duf, duf_cov, pen,
                       panel_label, save, style, wc)

style()
CLS = [  # palette key, duffing group stem, pendulum summary key
    ("lp", "mmn2", "MMN_dh2"), ("lp1", "mmn1", "MMN_dh1"), ("th", "mmn0", "obsTH(dh0)"), ("td", "td", "obsTD"),
    ("tdw", None, "obsTD_warm"), ("cfm", "cfm", "OT-CFM"), ("sb", "sb", None),
]


def tint(c, a):
    return tuple(a * np.array(to_rgb(c)) + (1 - a))


def big(ax, x, y, key, fill, ms=6.8, z=5):
    fc = {"A": PAL[key], "B": "white", "C": tint(PAL[key], 0.35)}[fill]
    ax.plot(x, y, MK[key], ms=ms, mfc=fc, mec=PAL[key] if fill != "A" else "0.0", mew=(1.3 if fill != "A" else 0.6),
            ls="", zorder=z)


def seeds(ax, a, b, key):
    ax.plot(a, b, MK[key], ms=2.6, mfc=tint(PAL[key], 0.6), mec="none", ls="", zorder=2)


fig, axs = plt.subplots(1, 3, figsize=(COLW, 1.98))
fig.subplots_adjust(left=0.075, right=0.995, top=0.77, bottom=0.20, wspace=0.30)

TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
inwin = TT + DD <= DUF_T + 1e-9
CA, CB = duf_cov(DUF_Z["XA0"]), duf_cov(DUF_Z["XB0"])
dAB = np.linalg.norm((CB - CA)[inwin]) / np.linalg.norm(CA[inwin])
dBA = np.linalg.norm((CA - CB)[inwin]) / np.linalg.norm(CB[inwin])


def mark_processes(ax, xa, yb, lim):
    ax.plot([0], [yb], "*", color="0.0", ms=10, clip_on=False, zorder=6)
    ax.text(0.06, yb + 0.10, r"A", ha="left", va="bottom", fontsize=8.5)
    ax.plot([xa], [0], "*", color="0.0", ms=10, clip_on=False, zorder=6)
    ax.text(xa + 0.12, 0.0, r"B", ha="left", va="center", fontsize=8.5)
    ax.plot([0, lim], [0, lim], color="0.7", lw=0.7, ls="--", zorder=-3)
    ax.set_xlim(-0.06, lim); ax.set_ylim(-0.06, 1.55)
    ax.set_xlabel(r"$\epsilon_C^{A}$", labelpad=1)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 0.5, 1, 1.5])


# (a) Duffing
ax = axs[0]
for key, dg, _ in CLS:
    if dg is None:
        continue
    for W in ("A", "B"):
        a, b = duf(f"{W}0_{dg}", "M2_in"), duf(f"{W}0_{dg}", "M2_in_vsB")
        if not len(a):
            continue
        if W == "A":
            seeds(ax, a, b, key)
        print(f"Duffing {W}-data {dg}: median eps_A {np.median(a):.3f} eps_B {np.median(b):.3f}")
        big(ax, np.median(a), np.median(b), key, W, ms=(7.6 if W == "A" else 5.4), z=(4 if W == "A" else 5))
mark_processes(ax, dAB, dBA, 2.2)
ax.set_ylabel(r"$\epsilon_C^{B}$", labelpad=1)
ax.text(0.97, 0.97, r"Duffing", transform=ax.transAxes, ha="right", va="top")
panel_label(ax, "a", dx=-0.24, dy=1.02)

# (b) pendulum
ax = axs[1]
CAp, CBp, CTp, CDp = PEN_ST["C_true"], PEN_ST["C_B"], PEN_ST["C_T"], PEN_ST["C_D"]
inp = (CTp[:, None] + CDp[None, :]) <= 15.0 + 1e-9
pAB = np.linalg.norm((CBp - CAp)[inp]) / np.linalg.norm(CAp[inp])
pBA = np.linalg.norm((CAp - CBp)[inp]) / np.linalg.norm(CBp[inp])
print("pendulum eps_C(B vs A) =", round(pAB, 3), " duffing =", round(dAB, 3))
for key, _, pk in CLS:
    if pk is None:
        continue
    a, b = pen(pk, "M2_in"), pen(pk, "M2_in_vs_worldB")
    seeds(ax, a, b, key)
    big(ax, np.median(a), np.median(b), key, "A")
mark_processes(ax, pAB, pBA, 2.4)
ax.text(0.97, 0.97, r"pendulum", transform=ax.transAxes, ha="right", va="top")
panel_label(ax, "b", dx=-0.17, dy=1.02)

# (c) process-C test
ax = axs[2]
E = WC_R["world_C"]["verification"]["epsC"]
SIZE = {"A": 8.0, "B": 6.3, "C": 4.6}            # nested sizes keep coincident medians visible
for key, dg, _ in CLS:
    if dg not in ("mmn2", "td", "cfm", "sb"):
        continue
    seeds(ax, wc(f"C0_{dg}", "dA"), wc(f"C0_{dg}", "dB"), key)
    for W in ("A", "B", "C"):
        a, b = wc(f"{W}0_{dg}", "dA"), wc(f"{W}0_{dg}", "dB")
        print(f"process-C test {W}-data {dg}: median eps_A {np.median(a):.3f} eps_B {np.median(b):.3f}"
              f" eps_C(.|C) {np.median(wc(f'{W}0_{dg}', 'dC')):.3f}")
        big(ax, np.median(a), np.median(b), key, W, ms=SIZE[W], z={"A": 4, "B": 5, "C": 6}[W])
mark_processes(ax, E["B_vs_A"], E["A_vs_B"], 2.2)
ax.plot([E["C_vs_A"]], [E["C_vs_B"]], "*", ms=11, mfc="white", mec="0.0", mew=1.0, zorder=6)
ax.text(E["C_vs_A"] + 0.12, E["C_vs_B"] + 0.04, r"C", ha="left", va="center", fontsize=8.5)
print("process C at eps_A, eps_B =", round(E["C_vs_A"], 3), round(E["C_vs_B"], 3))
ax.text(0.97, 0.97, r"Duffing", transform=ax.transAxes, ha="right", va="top")
h = [Line2D([], [], ls="", marker="o", ms=5, mfc=PAL["lp"], mec="0.0", mew=0.6, label=r"A data"),
     Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=PAL["lp"], mew=1.2, label=r"B data"),
     Line2D([], [], ls="", marker="o", ms=5, mfc=tint(PAL["lp"], 0.35), mec=PAL["lp"], mew=1.2, label=r"C data")]
ax.legend(handles=h, loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=7.6, handlelength=0.8,
          handletextpad=0.3, labelspacing=0.25, borderaxespad=0.1)
panel_label(ax, "c", dx=-0.17, dy=1.02)

order = ["lp", "lp1", "th", "td", "tdw", "sb", "cfm"]
h = [Line2D([], [], ls="", marker=MK[k], ms=5.5, mfc=PAL[k], mec="0.0", mew=0.5, label=LAB[k]) for k in order]
h.append(Line2D([], [], ls="", marker="*", ms=8, color="0.0", label=r"process"))
fig.legend(handles=h, loc="upper center", ncol=8, fontsize=7.6, handlelength=0.9, columnspacing=0.9,
           handletextpad=0.25, bbox_to_anchor=(0.53, 1.0))
save(fig, OUT / "fig2_prior_selects", close=True)
print("fig2_prior_selects (paper Fig. 3) written")
