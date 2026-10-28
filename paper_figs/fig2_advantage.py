"""Figure 2: the quantitative advantage of the lift--project SDE on both systems, at comparable marginal fit.

(a, b) Lagged correlation r(4, 4+d) = C(4,4+d) / sqrt(C(4,4) C(4+d,4+d)) versus lag d on the 0.1 grid, Duffing (a)
       and pendulum (b): truth (process A) black; lift--project seed median with the min-max range over its 5 seeds
       shaded; each baseline class as its seed median.
(c)    eps_C (relative two-time covariance error to process A) per class and seed, both systems; bar = seed median;
       dashed line = certified floor ||min(C*,0)||_F / ||C*||_F for every diffusion Markov in x (Prop. 2 / Cor. (i)).
(d)    eps_rho (lagged Spearman rank correlation error), same layout; floor from Cor. (ii).
(e)    training-window marginal fit W2_train per class and seed; dotted line = noise floor (independent truth sample).
Colour: author-approved exception to greyscale-first for the main figures (class palette in pf_common.PAL);
markers differ per class as well.
Sources:
  paper_figs/data/twotime_{duffing,pendulum}.npz   C_truth and C_<run> on the 0.1 grid (resim_twotime.py: stored
                                                         models re-simulated, no training; eps_C equals stored M2_in)
  duffing/results.json   groups A0_{mmn2,mmn0,td,sb,cfm}: M2_in, M0 per_seed
  pendulum/gate.json     summary {MMN_dh2, obsTH(dh0), obsTD, obsTD_warm, OT-CFM}: M2_in, M0 per_seed
  monotone_stat/results_{duffing,pendulum}.json   groups.<g>.eps_rho_vsA.per_seed; worlds.A.lower_bound_cov
                              (0.729 Duffing, 0.737 pendulum) and lower_bound_rho (0.719, 0.695)
  noise floors of W2_train: duffing/REPORT.txt sec. 1 (0.009), pendulum/REPORT.txt sec. 2 (0.014)
usage: python fig2_advantage.py   -> out/fig2_advantage.{png,eps}
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from pf_common import COLW, HERE, L4DC, LAB, MK, OUT, PAL, duf, panel_label, pen, save, style

style()
MON = {s: json.loads((L4DC / "monotone_stat" / f"results_{s}.json").read_text()) for s in ("duffing", "pendulum")}
TW = {s: np.load(HERE / "data" / f"twotime_{s}.npz") for s in ("duffing", "pendulum")}
NOISE_W2 = {"duffing": 0.009, "pendulum": 0.014}

# class key -> (duffing group, pendulum group / gate.json key)
SYSC = {
    "duffing": [("lp", "A0_mmn2"), ("th", "A0_mmn0"), ("td", "A0_td"), ("sb", "A0_sb"), ("cfm", "A0_cfm")],
    "pendulum": [("lp", "mmn_dh2"), ("th", "mmn_dh0"), ("td", "td"), ("tdw", "tdw"), ("cfm", "cfm")],
}
PKEY = {"mmn_dh2": "MMN_dh2", "mmn_dh0": "obsTH(dh0)", "td": "obsTD", "tdw": "obsTD_warm", "cfm": "OT-CFM"}


def tint(c, a):
    return tuple(a * np.array(to_rgb(c)) + (1 - a))


def metric(sysname, g, m):
    if m == "rho":
        return np.array(MON[sysname]["groups"][g]["eps_rho_vsA"]["per_seed"])
    if sysname == "duffing":
        return duf(g, m)
    return pen(PKEY[g], m)


fig = plt.figure(figsize=(COLW, 3.15))
gs0 = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45, left=0.085, right=0.995, top=0.87, bottom=0.085)
gtop = gs0[0].subgridspec(1, 2, wspace=0.22)
gbot = gs0[1].subgridspec(1, 3, wspace=0.50)

# ------------------------------------------------------------------ (a, b) lagged correlation at t = 4
T0 = 4.0
for j, sysname in enumerate(("duffing", "pendulum")):
    ax = fig.add_subplot(gtop[0, j])
    Z = TW[sysname]
    t, d = Z["t"], Z["d"]
    i0 = int(round(T0 / 0.1))

    def corr(C):
        return C[i0] / np.sqrt(C[i0, 0] * C[i0 + np.arange(len(d)), 0])

    ax.axhline(0, color="0.6", lw=0.6, zorder=-5)
    for key, g in SYSC[sysname][::-1]:
        R = np.array([corr(Z[f"C_{g}_s{s}"]) for s in range(5)])
        med = np.median(R, 0)
        if key == "lp":
            ax.fill_between(d, R.min(0), R.max(0), color=tint(PAL[key], 0.25), lw=0, zorder=1)   # solid tint: EPS has no alpha
            ax.plot(d, med, color=PAL[key], lw=2.0, zorder=4)
        else:
            ax.plot(d, med, color=PAL[key], lw=1.2, zorder=2, ls=("--" if key in ("th", "tdw") else "-"))
        print(f"{sysname} {g}: min over d of median r(4,4+d) = {med.min():.3f}")
    rt = corr(Z["C_truth"])
    ax.plot(d, rt, color="0.0", lw=1.3, ls=(0, (4, 1.5)), zorder=5)
    print(f"{sysname} truth: min r(4,4+d) = {rt.min():.3f}")
    ax.set_xlim(0, 8); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$d$", labelpad=1)
    if j == 0:
        ax.set_ylabel(r"$r(4,4{+}d)$")
    ax.text(0.5, 1.02, (r"Duffing" if j == 0 else r"pendulum"), transform=ax.transAxes, ha="center", va="bottom")
    panel_label(ax, "ab"[j], dx=(-0.13 if j == 0 else -0.08), dy=1.02)

# ------------------------------------------------------------------ (c, d, e) per-class strips, both systems
FLOOR = {"C": {s: MON[s]["worlds"]["A"]["lower_bound_cov"] for s in MON},
         "rho": {s: MON[s]["worlds"]["A"]["lower_bound_rho"] for s in MON}}
print("floors:", {k: {s: round(v, 3) for s, v in f.items()} for k, f in FLOOR.items()})
PANELS = [("M2_in", r"$\epsilon_C$", "C", (0.08, 4.0)), ("rho", r"$\epsilon_\rho$", "rho", (0.12, 3.0)),
          ("M0", r"$\overline{W}_2^{\,\mathrm{train}}$", None, (0.004, 1.6))]
rng = np.random.default_rng(1)
TICKS = {"C": [0.1, 0.2, 0.5, 1, 2], "rho": [0.2, 0.5, 1, 2]}
for j, (m, ylab, fl, ylim) in enumerate(PANELS):
    ax = fig.add_subplot(gbot[0, j])
    for si, sysname in enumerate(("duffing", "pendulum")):
        x0 = si * 5.8
        for ci, (key, g) in enumerate(SYSC[sysname]):
            v = metric(sysname, g, m)
            x = x0 + ci
            jit = rng.uniform(-0.18, 0.18, v.size)
            ax.plot(x + jit, v, MK[key], ms=(3.6 if key == "lp" else 3.0), mfc=PAL[key], mec="white", mew=0.35,
                    ls="", zorder=3)
            ax.plot([x - 0.36, x + 0.36], [np.median(v)] * 2, "-", color=PAL[key], lw=(2.2 if key == "lp" else 1.6),
                    zorder=4, solid_capstyle="butt")
            if m in ("M2_in", "rho"):
                print(f"{m:5s} {sysname:8s} {g:8s} median {np.median(v):.3f}  [min {v.min():.3f}, max {v.max():.3f}]")
        xs = (x0 - 0.5, x0 + len(SYSC[sysname]) - 0.5)
        if fl is not None:
            f = FLOOR[fl][sysname]
            ax.plot(xs, [f, f], ls="--", color="0.0", lw=0.9, zorder=2)
            ax.fill_between(xs, [f, f], [ylim[1]] * 2, color="0.93", lw=0, zorder=-6)
        else:
            nf = NOISE_W2[sysname]
            ax.plot(xs, [nf, nf], ls=":", color="0.0", lw=1.0, zorder=2)
    ax.set_yscale("log"); ax.set_ylim(*ylim); ax.set_xlim(-0.7, 10.5)
    ax.set_xticks([2.0, 7.8]); ax.set_xticklabels([r"Duffing", r"pendulum"])
    ax.tick_params(axis="x", which="both", top=False, bottom=False, pad=2)
    ax.axvline(4.9, color="0.75", lw=0.6)
    ax.set_ylabel(ylab, labelpad=(4 if fl is None else 2))
    panel_label(ax, "cde"[j], dx=-0.30, dy=1.02)
    if fl is not None:
        ax.text(2.7, FLOOR[fl]["duffing"] * 0.93, r"floor" "\n" r"(Cor.~4)", ha="center", va="top", fontsize=7.6)
        ax.set_yticks(TICKS[fl]); ax.set_yticklabels([rf"${v:g}$" for v in TICKS[fl]]); ax.minorticks_off()
    else:
        ax.text(2.0, NOISE_W2["duffing"] * 0.85, r"noise floor", ha="center", va="top", fontsize=7.6)

# ------------------------------------------------------------------ class legend (shared)
order = ["lp", "th", "td", "tdw", "sb", "cfm"]
h = [Line2D([], [], color="0.0", lw=1.3, ls=(0, (4, 1.5)), label=r"truth")]
h += [Line2D([], [], color=PAL[k], lw=(2.0 if k == "lp" else 1.2), ls=("--" if k in ("th", "tdw") else "-"),
             marker=MK[k], ms=4, mfc=PAL[k], mec="white", mew=0.35, label=LAB[k]) for k in order]
fig.legend(handles=h, loc="upper center", ncol=7, fontsize=7.6, handlelength=1.9, columnspacing=0.9,
           handletextpad=0.4, bbox_to_anchor=(0.54, 1.005))
save(fig, OUT / "fig2_advantage", close=True)
print("fig2_advantage written")
