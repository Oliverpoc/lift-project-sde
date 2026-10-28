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

fig, axs = plt.subplots(1, 3, figsize=(COLW, 1.62), gridspec_kw=dict(width_ratios=[0.82, 1.0, 1.0]))
fig.subplots_adjust(left=0.085, right=0.995, top=0.80, bottom=0.2, wspace=0.40)

ax = axs[0]
ax.plot(TK, DUF_DIST["W2_A1_vs_B1_at_tk"], "o-", color=DC, ms=3.5, lw=1.4, label=r"$\mu_1$: A vs B")
ax.plot(TK, DUF_DIST["W2_A0_vs_B0_at_tk"], "s--", color="0.45", ms=3.0, lw=1.0, mfc="white", label=r"$\mu_0$: A vs B")
ax.plot(TK, DUF_DIST["W2_A1_vs_A1b_noise_at_tk"], ":", color="0.0", lw=1.2, label=r"noise floor")
ax.set_yscale("log"); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$W_2(p^A_t,p^B_t)$"); ax.set_xlim(-0.5, 15.5)
ax.set_ylim(1e-8, 2.0); ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.07), fontsize=7.6, handlelength=2.0, borderaxespad=0.1, labelspacing=0.2)
print("max W2 mu1:", round(DUF_DIST["W2_A1_vs_B1_max"], 3), "noise max:", round(DUF_DIST["W2_A1_vs_A1b_noise_max"], 4),
      "ratio:", round(DUF_DIST["W2_A1_vs_B1_max"] / DUF_DIST["W2_A1_vs_A1b_noise_max"], 1))
panel_label(ax, "a", dx=-0.3, dy=1.02)

SHR = L4DC / "shared"
SHR_R = json.loads((SHR / "results.json").read_text())
SHR_V = json.loads((SHR / "verdict.json").read_text())
CLS = [("MMN", "mmn2", "mmn2", "mmnc2", r"lift--project", "lp"), ("obs-TD", "td", "tdsh", "tdlab", r"obs-TD", "td"),
       ("SB", "sb", "sbsh", "sb", r"SB", "sb")]
# training condition: (key, legend label, face (open / class colour / light tint), x offset within a class)
CONDS = [("mu0", r"$\mu_0$ only", "open", -0.27), ("sh", r"$\mu_0{+}\mu_1$, common dynamics", "full", 0.0),
         ("c", r"$\mu_0{+}\mu_1$, per-experiment dynamics", "tint", 0.27)]


def shr(tag, key):
    return np.array([json.loads((SHR / "runs" / f"{tag}_s{s}" / "eval.json").read_text())[key] for s in range(5)])


def face(kind, c):
    return {"open": "white", "full": c, "tint": tint(c, 0.35)}[kind]


rng = np.random.default_rng(0)
axb, axc = axs[1], axs[2]
for i, (cl, t0, tsh, tc, lab, pk) in enumerate(CLS):
    col = PAL[pk]
    tags = {"mu0": f"A0_{t0}", "mu1": f"A1_{t0}", "sh": f"A01_{tsh}", "c": f"A01_{tc}"}
    ref0, ref1 = shr(tags["mu0"], "M0_0").mean(), shr(tags["mu1"], "M0_1").mean()
    for key, clab, fk, dx in CONDS:
        x = i + dx
        # (c) lagged-covariance error under mu0: seeds (jittered) and seed median
        v = shr(tags[key], "epsC_0")
        assert abs(v.mean() - SHR_R[f"{cl}|{key}"]["epsC_0_mean"]) < 1e-12
        axc.bar(x, np.median(v), width=0.25, color=face(fk, col), edgecolor=col, lw=0.9, zorder=2)
        axc.plot(x + rng.uniform(-0.06, 0.06, v.size), v, "o", ms=1.9, mfc="0.15", mec="none", ls="", zorder=3)
        if key == "mu0":
            continue
        # (b) marginal fit with two experiments relative to single-experiment training, worse of the two experiments
        r0 = shr(tags[key], "M0_0").mean() / ref0; r1 = shr(tags[key], "M0_1").mean() / ref1
        assert ((r0 <= 1.5) and (r1 <= 1.5)) == SHR_V["joint"][f"{cl}|{key}"]["fit_comparable"]
        r = max(r0, r1)
        xb = i + (-0.17 if key == "sh" else 0.17)
        axb.bar(xb, r, width=0.3, color=face(fk, col), edgecolor=col, lw=0.9)
        print(f"{lab:7s} {key:3s} eps_C median {np.median(v):.3f}  W2 ratio mu0 {r0:.2f} mu1 {r1:.2f} -> {r:.2f}")

axb.axhline(1.0, color="0.0", lw=0.8, ls=":")
axb.set_ylim(0, 4.0); axb.set_yticks([0, 1, 2, 3, 4])
axb.set_ylabel(r"$\overline W_2^{\,\mathrm{train}}$ ratio")
for ax in (axb, axc):
    ax.set_xticks(range(len(CLS))); ax.set_xticklabels([c[4] for c in CLS], fontsize=7.0)
    for tl, c in zip(ax.get_xticklabels(), CLS):
        tl.set_color(PAL[c[5]])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlim(-0.5, 2.5)
# error floor of Corollary 4 for every diffusion Markov in x (Duffing, mu_0); grey: values such models can reach
FL = json.loads((L4DC / "monotone_stat" / "results_duffing.json").read_text())["worlds"]["A"]["lower_bound_cov"]
print("Duffing floor", round(FL, 3))
axc.axhline(FL, color="0.0", lw=0.9, ls="--", zorder=1)
axc.axhspan(FL, 1.6, color="0.93", lw=0, zorder=-6)
axc.text(-0.45, FL + 0.03, r"floor", ha="left", va="bottom", fontsize=7.4)
axc.set_ylim(0.0, 1.6); axc.set_yticks([0, 0.5, 1, 1.5])
axc.set_ylabel(r"$\epsilon_C$ under $\mu_0$")
panel_label(axb, "b", dx=-0.24, dy=1.02)
panel_label(axc, "c", dx=-0.24, dy=1.02)
h = [Line2D([], [], ls="", marker="s", ms=5, mfc=face(fk, "0.3"), mec="0.3", mew=0.9, label=clab) for _, clab, fk, _ in CONDS]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.8, handlelength=1.0,
           handletextpad=0.3, columnspacing=1.6, borderaxespad=0.0)
save(fig, OUT / "fig4_multi_experiment", close=True)
print("fig4 written")
