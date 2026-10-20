"""Figure 1 (hero): the lift--project SDE and the two-time covariance it recovers from unpaired snapshots.

(a) Cartoon of unpaired snapshots: samples at t0, t1, t2 with no known correspondence between times.
(b) Cloud-projection schematic of the method (after Fig. 1 of the SIAM UQ paper), drawn with a toy damped-rotation
    SDE in z = (x, h): lift x0 -> z0, one autonomous SDE moves the cloud, projection pi_x gives the marginals that the
    W2 loss compares with the data.  The toy SDE is illustrative only; no stored result is used in (a) or (b).
(c) Two-time covariance C(t, t+d) on the 0.1 sampling grid (t in [0, 15], d in [0, 8]) for the true Duffing
    system (process A) and for one trained model of each of three classes (the seed whose eps_C is the seed median),
    on one RdBu colour scale centred at zero.  Black line: C = 0.  Dashed line: t + d = T (eps_C uses t + d <= T).
    The truth has negative bands; lift--project reproduces them; the observed-state models stay >= 0 (Prop. 2).
Colour: author-approved exception to greyscale-first for the main figures (class palette in pf_common.PAL).
Sources:
  paper_figs/data/twotime_duffing.npz       C_truth (from duffing/data/truth.npz XA0) and C_<run> of every stored
                                                 model, re-simulated without training by resim_twotime.py
  duffing/results.json                      M2_in per seed (eps_C printed on the panels; equals the re-simulated value)
usage: python fig1_problem.py   -> out/fig1_problem.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pf_common import COLW, DUF, HERE, LAB, OUT, PAL, duf, panel_label, save, style

style()
# Times text and maths through LaTeX, to match the paper body (\usepackage{times})
plt.rcParams.update({"text.usetex": True, "font.serif": ["Times"],
                     "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"
                                            r"\usepackage{newtxtext}\usepackage{newtxmath}"})
T = 15.0
Z = np.load(HERE / "data" / "twotime_duffing.npz")
t, d = Z["t"], Z["d"]


def median_seed(group):
    v = duf(group, "M2_in")
    s = int(np.argsort(v)[len(v) // 2])
    assert abs(float(Z[f"epsC_{group}_s{s}"]) - v[s]) < 1e-6
    return s, v[s]


fig = plt.figure(figsize=(COLW, 3.05))
gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.05], width_ratios=[1, 1, 1, 1, 0.055], hspace=0.33, wspace=0.10,
                      left=0.065, right=0.925, top=0.995, bottom=0.11)
gtop = gs[0, :].subgridspec(1, 2, width_ratios=[1.0, 2.75], wspace=0.10)
FS = 7.6
C0, CK = "#2B6CB0", "#E07B1A"           # snapshot colours: first time (blue), later time (orange), as in the SIAM figure

# ============================================================================ (a) unpaired snapshots (cartoon)
ax = fig.add_subplot(gtop[0])
rng = np.random.default_rng(3)
cols = [C0, "#8C6BB1", CK]
ys = [np.array([-1.7, -1.3, -1.0, -0.75, -0.45, -0.1, 0.3]), np.array([-1.0, -0.6, -0.3, 0.05, 0.4, 0.75, 1.15]),
      np.array([-0.2, 0.25, 0.6, 0.9, 1.2, 1.5, 1.85])]
for k, y in enumerate(ys):
    ax.scatter(np.full(y.size, k) + rng.uniform(-0.10, 0.10, y.size), y, s=14, color=cols[k], lw=0, zorder=3)
i0 = int(np.argmax(ys[0]))
p0 = (0.0, ys[0][i0])
ax.scatter(*p0, s=34, facecolor="none", edgecolor="0.0", lw=0.9, zorder=4)
for j in (1, 3, 5):
    ax.plot([p0[0], 1.0], [p0[1], ys[1][j]], ls=(0, (2, 1.5)), color="0.35", lw=0.7, zorder=2)
    ax.text(1.17, ys[1][j], r"?", fontsize=FS, ha="left", va="center")
ax.set_xlim(-0.45, 2.45); ax.set_ylim(-2.1, 2.4); ax.minorticks_off()
ax.tick_params(top=False, right=False)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$t_0$", r"$t_1$", r"$t_2$"]); ax.set_yticks([])
ax.set_ylabel(r"$x$", labelpad=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(1.0, 2.4, r"unpaired snapshots", ha="center", va="top", fontsize=FS)
panel_label(ax, "a", dx=-0.17, dy=0.93)

# ============================================================================ (b) lift -> autonomous SDE in z -> project
ax = fig.add_subplot(gtop[1])
ax.axis("off")
A, B, YP = 0.55, 0.50, 1.55             # oblique view: screen = (x + A h, YP + B h)


def scr(x, h):
    return x + A * h, YP + B * h


# a toy latent SDE (damped rotation in (x, h)); only the shape matters for the schematic
rng = np.random.default_rng(1)
n, dt, nst = 500, 0.02, 157             # nst * dt = pi: half a turn
x = rng.normal(-1.15, 0.20, n); h = rng.normal(0.0, 0.22, n)
P = np.empty((nst + 1, n, 2)); P[0, :, 0], P[0, :, 1] = x, h
for i in range(nst):
    x, h = x + h * dt, h + (-x - 0.08 * h) * dt + 0.16 * np.sqrt(dt) * rng.standard_normal(n)
    P[i + 1, :, 0], P[i + 1, :, 1] = x, h

# latent plane and its axes
XL, HL = (-2.0, 2.0), (-1.45, 1.45)
corn = [scr(XL[0], HL[0]), scr(XL[1], HL[0]), scr(XL[1], HL[1]), scr(XL[0], HL[1])]
ax.add_patch(plt.Polygon(corn, closed=True, fc="#F3F1EC", ec="0.55", lw=0.7, zorder=0))
o = scr(XL[0], HL[0])
ax.add_patch(FancyArrowPatch(o, scr(XL[1] + 0.25, HL[0]), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.2"))
ax.add_patch(FancyArrowPatch(o, scr(XL[0], HL[1] + 0.35), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.2"))
ax.text(*scr(XL[1] + 0.32, HL[0]), r"$x$ (observed)", fontsize=FS, ha="left", va="center")
ax.text(*scr(XL[0] - 0.08, HL[1] + 0.42), r"$h$ (hidden)", fontsize=FS, ha="right", va="bottom")

# sample paths, coloured from the first to the last time
from matplotlib.colors import LinearSegmentedColormap
TC = LinearSegmentedColormap.from_list("tk", [C0, "#8C6BB1", CK])
for j in rng.choice(n, 7, replace=False):
    sx, sy = scr(P[:, j, 0], P[:, j, 1])
    for i in range(0, nst, 3):
        ax.plot(sx[i:i + 4], sy[i:i + 4], ls="-", color=TC(i / nst), lw=0.6, alpha=0.85, zorder=2)
# clouds at t0 and tk
for i, c in ((0, C0), (nst, CK)):
    sx, sy = scr(P[i, :, 0], P[i, :, 1])
    ax.scatter(sx, sy, s=1.2, color=c, lw=0, alpha=0.7, zorder=3, rasterized=True)
ax.text(*scr(-1.25, 0.95), r"$z_0$", fontsize=FS, ha="center", va="center", color=C0)
ax.text(*scr(2.3, -0.25), r"$z(t_k)$", fontsize=FS, ha="left", va="center", color=CK)
ax.text(*scr(0.55, 2.15), r"\textbf{autonomous SDE}: $\mathrm{d}z=f_\theta(z)\,\mathrm{d}t+g_\theta(z)\,\mathrm{d}W$",
        fontsize=FS, ha="center", va="bottom", color=PAL["lp"])

# observed axis below with the projected marginals and the data
Y0, HD = -0.55, 0.62
ax.plot([-2.2, 2.4], [Y0, Y0], ls="-", color="0.2", lw=0.8)
ax.text(2.45, Y0, r"$x$", fontsize=FS, ha="left", va="center")
g = np.linspace(-2.1, 2.2, 300)
for i, c, m in ((0, C0, -1.15), (nst, CK, None)):
    v = P[i, :, 0]
    kde = np.exp(-0.5 * ((g[:, None] - v[None]) / 0.12) ** 2).sum(1)
    kde = HD * kde / kde.max()
    ax.fill_between(g, Y0, Y0 + kde, color=c, alpha=0.35, lw=0, zorder=2)
    ax.plot(g, Y0 + kde, ls="-", color=c, lw=0.8, zorder=2)
    dat = rng.choice(v, 30, replace=False)
    ax.plot(dat, np.full(dat.size, Y0 - 0.09), ls="none", marker="|", ms=3.5, mew=0.6, color="0.15", zorder=3)
# lift (up, at t0) and project (down, at tk)
xa, xb = -1.15, 1.10
ax.add_patch(FancyArrowPatch((xa, Y0 + HD + 0.08), scr(xa, -0.45), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                             color=C0, zorder=4))
ax.text(xa - 0.08, 0.32, r"lift $h_0=G_\phi(x_0,\varepsilon)$", fontsize=FS, ha="right", va="center", color=C0)
sx, sy = scr(xb, -0.45)
ax.add_patch(FancyArrowPatch((sx, sy), (xb, Y0 + HD + 0.08), arrowstyle="-|>", mutation_scale=8, lw=1.0,
                             ls=(0, (3, 1.5)), color=CK, zorder=4))
ax.text(xb + 0.42, 0.32, r"project $\pi_x$", fontsize=FS, ha="left", va="center", color=CK)
ax.text(0.1, Y0 - 0.24, r"loss: $\sum_k W_2^2$(projected marginal, snapshot data $|$); $h$ never enters it",
        fontsize=FS, ha="center", va="top")
ax.set_xlim(-2.75, 3.05); ax.set_ylim(-1.0, 3.25)
panel_label(ax, "b", dx=0.02, dy=0.93)

# ============================================================================ (c) two-time covariance heat maps
vmax = np.abs(Z["C_truth"]).max()
norm = TwoSlopeNorm(0.0, -vmax, vmax)
panels = [("truth", r"truth (process A)", Z["C_truth"], None, "truth")]
for g, key in (("A0_mmn2", "lp"), ("A0_td", "td"), ("A0_cfm", "cfm")):
    s, e = median_seed(g)
    panels.append((g, LAB[key], Z[f"C_{g}_s{s}"], e, key))
    print(f"{g}: median seed s{s}, eps_C = {e:.3f}, min C = {Z[f'C_{g}_s{s}'].min():.4f}")
print(f"truth: C range [{Z['C_truth'].min():.3f}, {Z['C_truth'].max():.3f}]")
for j, (g, name, C, e, key) in enumerate(panels):
    ax = fig.add_subplot(gs[1, j])
    pm = ax.pcolormesh(t, d, C.T, cmap="RdBu_r", norm=norm, shading="nearest", rasterized=True)
    if C.min() < -0.01:                     # zero contour only where C has clearly negative values
        ax.contour(t, d, C.T, levels=[0.0], colors="0.0", linewidths=0.6)
    ax.plot([T - 8, T], [8, 0], ls="--", color="0.0", lw=0.7)
    ax.set_xlim(0, T); ax.set_ylim(0, 8)
    ax.set_xticks([0, 5, 10, 15] if j == 3 else [0, 5, 10]); ax.set_yticks([0, 4, 8])
    ax.set_xlabel(r"$t$", labelpad=1)
    if j == 0:
        ax.set_ylabel(r"$d$", labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.tick_params(which="both", direction="out", top=False, right=False)
    for sp in ax.spines.values():
        sp.set_edgecolor(PAL[key]); sp.set_linewidth(1.6 if key == "lp" else 1.0)
    txt = name if e is None else name + rf", $\epsilon_C{{=}}{e:.2f}$"
    ax.text(0.0, 1.035, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6,
            color=("0.0" if key == "truth" else PAL[key]))
    if j == 0:
        panel_label(ax, "c", dx=-0.27, dy=1.03)
cax = fig.add_subplot(gs[1, 4])
cb = fig.colorbar(pm, cax=cax, extend="max")
cb.set_label(r"$C(t,t{+}d)$", labelpad=2); cb.ax.tick_params(direction="in", labelsize=7.6)

save(fig, OUT / "fig1_problem", close=True)
print("fig1 written")
