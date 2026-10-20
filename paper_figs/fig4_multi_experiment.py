"""Figure 4: multiple initial distributions (Duffing).

Colour (author-approved for the main figures): (b) uses the class palette of pf_common.PAL (colour and marker =
class; fill = training condition: open = mu0 only, full = shared, light = conditioned); (a) draws the mu1 A-B
separation in the data colour of Fig. 1.

(a) marginal distance W2 between process A and process B at each snapshot time, under mu0 (identical
    marginals by construction) and under mu1 (B extended as the unique x-Markov process), with the
    finite-sample noise floor (A vs an independent A sample, under mu1).
(b) controlled shared vs experiment-conditioned comparison (shared, pre-registered): eps_C under mu0 for mu0 only / mu0+mu1 shared / mu0+mu1 conditioned, per class
    (lifted, obs-TD, SB), 5 seeds each (all shown), bar = seed median (paper-wide summary; the
    pre-specified criteria themselves use means, reported in the paper appendix). A dagger marks joint models that failed the
    pre-registered marginal-fit criterion (a): M0 > 1.5 x the single-experiment run of the same class.
(An earlier panel with the uncontrolled duffing mu0 -> mu0+mu1 runs was dropped: the same runs reappear in (b);
 OT-CFM, not part of the controlled study, is reported in the paper text only.)
Sources: duffing/data/distinguishability.json; shared/runs/<tag>/eval.json (per seed), cross-checked against shared/results.json and verdict.json.
usage: python fig4_multi_experiment.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from pf_common import COLW, DUF_DIST, L4DC, MK, OUT, PAL, panel_label, save, style

style()
TK = np.arange(16.0)
DC = "#21918C"   # data colour of Fig. 1 (viridis teal): the A-B separation is a property of the data


def tint(c, a):
    return tuple(a * np.array(to_rgb(c)) + (1 - a))

fig, axs = plt.subplots(1, 2, figsize=(0.85 * COLW, 1.48), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
fig.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.27, wspace=0.32)  # drawn at printed width (0.85 textwidth)

ax = axs[0]
ax.plot(TK, DUF_DIST["W2_A1_vs_B1_at_tk"], "o-", color=DC, ms=3.5, lw=1.4, label=r"$\mu_1$: A vs B")
ax.plot(TK, DUF_DIST["W2_A0_vs_B0_at_tk"], "s--", color="0.45", ms=3.0, lw=1.0, mfc="white", label=r"$\mu_0$: A vs B")
ax.plot(TK, DUF_DIST["W2_A1_vs_A1b_noise_at_tk"], ":", color="0.0", lw=1.2, label=r"noise floor")
ax.set_yscale("log"); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$W_2(p^A_t,p^B_t)$"); ax.set_xlim(-0.5, 15.5)
ax.set_ylim(1e-8, 2.0); ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.07), fontsize=7.6, handlelength=2.0, borderaxespad=0.1, labelspacing=0.2)
print("max W2 mu1:", round(DUF_DIST["W2_A1_vs_B1_max"], 3), "noise max:", round(DUF_DIST["W2_A1_vs_A1b_noise_max"], 4),
      "ratio:", round(DUF_DIST["W2_A1_vs_B1_max"] / DUF_DIST["W2_A1_vs_A1b_noise_max"], 1))
panel_label(ax, "a", dx=-0.25, dy=1.02)

ax = axs[1]
SHR = L4DC / "shared"
SHR_R = json.loads((SHR / "results.json").read_text())
SHR_V = json.loads((SHR / "verdict.json").read_text())
CLS = [("MMN", "mmn2", "mmn2", "mmnc2", r"lift--project", "lp"), ("obs-TD", "td", "tdsh", "tdlab", r"obs-TD", "td"),
       ("SB", "sb", "sbsh", "sb", r"SB", "sb")]
# condition: (key, label, face (open / class colour / light tint), x offset); the class sets colour and marker shape
CONDS = [("mu0", r"$\mu_0$ only", "open", -0.26), ("sh", r"$\mu_0{+}\mu_1$ shared", "full", 0.0),
         ("c", r"$\mu_0{+}\mu_1$ conditioned", "tint", 0.26)]


def shr(tag, key):
    return np.array([json.loads((SHR / "runs" / f"{tag}_s{s}" / "eval.json").read_text())[key] for s in range(5)])


def face(kind, c):
    return {"open": "white", "full": c, "tint": tint(c, 0.35)}[kind]


for i, (cl, t0, tsh, tc, lab, pk) in enumerate(CLS):
    col = PAL[pk]
    tags = {"mu0": f"A0_{t0}", "mu1": f"A1_{t0}", "sh": f"A01_{tsh}", "c": f"A01_{tc}"}
    ref0, ref1 = shr(tags["mu0"], "M0_0").mean(), shr(tags["mu1"], "M0_1").mean()
    for key, clab, fk, dx in CONDS:
        v = shr(tags[key], "epsC_0")
        assert abs(v.mean() - SHR_R[f"{cl}|{key}"]["epsC_0_mean"]) < 1e-12
        x = i + dx
        ax.plot(x + 0 * v, v, MK[pk], ms=3.6, mfc=face(fk, col), mec=col, mew=0.9, ls="")
        ax.plot([x - 0.11, x + 0.11], [np.median(v)] * 2, "-", color=col, lw=1.8, solid_capstyle="butt")
        if key in ("sh", "c"):
            r0 = shr(tags[key], "M0_0").mean() / ref0; r1 = shr(tags[key], "M0_1").mean() / ref1
            ok = (r0 <= 1.5) and (r1 <= 1.5)
            assert ok == SHR_V["joint"][f"{cl}|{key}"]["fit_comparable"]
            if not ok:
                ax.text(x, v.max() + 0.04, r"$\dagger$", ha="center", va="bottom", fontsize=10, color=col)
            print(f"{lab:7s} {key:3s} eps_C(mu0) median {np.median(v):.3f} mean {v.mean():.3f}  M0 ratio (means) {r0:.2f}/{r1:.2f}  fit-comparable {ok}")
        else:
            print(f"{lab:7s} mu0 eps_C(mu0) median {np.median(v):.3f} mean {v.mean():.3f}")
ax.set_xticks(range(len(CLS))); ax.set_xticklabels([c[4] for c in CLS])
for tl, c in zip(ax.get_xticklabels(), CLS):
    tl.set_color(PAL[c[5]])
ax.tick_params(axis="x", which="both", top=False, bottom=False)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.0, 2.45); ax.set_yticks([0, 0.5, 1, 1.5, 2])
ax.set_ylabel(r"$\epsilon_C$ under $\mu_0$")
h = [Line2D([], [], ls="", marker="o", ms=4, mfc=face(fk, "0.25"), mec="0.25", mew=0.9, label=clab) for _, clab, fk, _ in CONDS]
ax.legend(handles=h, loc="upper left", fontsize=7.6, handlelength=1.0, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
panel_label(ax, "b", dx=-0.17, dy=1.02)
save(fig, OUT / "fig4_multi_experiment", close=True)
print("fig4 written")
