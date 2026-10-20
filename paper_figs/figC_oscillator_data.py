"""Appendix data figures for the two oscillators (process A): Duffing and damped pendulum.

One figure per system, same layout:
  (a) sample paths x(t) of the true system in the training window, coloured by the hidden velocity;
  (b) phase portrait (x, hidden velocity) of three paths, coloured by time;
  (c) the snapshot marginals p_{t_k}: kernel density estimates of x at the 16 training times (viridis by t)
      and, below a separator, at the extrapolation times 1.5T, 2T, 3T (dashed outlines);
  (d) the unpaired training data as the model sees it: (t_k, x) with horizontal jitter.
Sources (nothing is re-simulated; same parameters and seeds as in the paper):
  Duffing   duffing/data/truth.npz   (tg, XA0, VA0: 4000 evaluation paths of process A, written by duffing/prep.py)
            duffing/data/snap_A0.npy (16 x 2000 unpaired training snapshots, process A, mu_0; duffing/common.py)
  Pendulum  pendulum/truth.npz       (tg, Xt, Vt: 4000 evaluation paths, written by pendulum/evaluate.py truth)
            pendulum/data_train.npy  (16 x 2000 unpaired training snapshots; pendulum/common.py)
usage: python figC_oscillator_data.py   -> out/figC_duffing_data.{png,eps}, out/figC_pendulum_data.{png,eps}
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from figC_data_common import VIR, colored_line, rank_picks
from pf_common import COLW, L4DC, OUT, panel_label, save, style

style()
T, DT = 15.0, 0.1
TK = np.arange(16, dtype=float)
TEX = [1.5 * T, 2 * T, 3 * T]
idx = lambda t: int(round(t / DT))

SYSTEMS = {
    "duffing": dict(truth=L4DC / "duffing/data/truth.npz", X="XA0", V="VA0",
                    snap=L4DC / "duffing/data/snap_A0.npy", vsym=r"v", name=r"Duffing", out="figC_duffing_data"),
    "pendulum": dict(truth=L4DC / "pendulum/truth.npz", X="Xt", V="Vt",
                     snap=L4DC / "pendulum/data_train.npy", vsym=r"\omega", name=r"pendulum", out="figC_pendulum_data"),
}


def make(key):
    S = SYSTEMS[key]
    Z = np.load(S["truth"])
    tg, X, V = Z["tg"], Z[S["X"]], Z[S["V"]]
    snap = np.load(S["snap"])
    assert snap.shape == (16, 2000)
    m = tg <= T + 1e-9
    vs = S["vsym"]
    print(f"[{key}] truth {X.shape}, snapshots {snap.shape}, x range in window "
          f"[{X[m].min():.2f}, {X[m].max():.2f}], {vs} range [{V[m].min():.2f}, {V[m].max():.2f}]")

    fig = plt.figure(figsize=(COLW, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42, left=0.08, right=0.97, top=0.95, bottom=0.09)
    ylim = (X[m].min() - 0.15, X[m].max() + 0.15)

    # (a) sample paths coloured by the hidden velocity
    ax = fig.add_subplot(gs[0, 0])
    q = np.quantile(X[m], [0.05, 0.95], axis=1)
    ax.fill_between(tg[m], q[0], q[1], color="0.9", lw=0, zorder=-6)
    picks, _ = rank_picks(X[0])
    vmax = np.quantile(np.abs(V[m][:, picks]), 0.99)
    nv = Normalize(-vmax, vmax)
    for i in picks:
        lc = colored_line(ax, tg[m], X[m][:, i], V[m][:, i], VIR, nv, lw=1.2)
    ax.set_xlim(0, T); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x(t)$")
    cb = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(rf"${vs}(t)$"); cb.ax.tick_params(direction="in")
    panel_label(ax, "a", dx=-0.17, dy=1.02)

    # (b) phase portrait coloured by time
    ax = fig.add_subplot(gs[0, 1])
    nt = Normalize(0, T)
    for i in picks[1:4]:
        lc = colored_line(ax, X[m][:, i], V[m][:, i], tg[m], VIR, nt, lw=1.0)
        ax.plot(X[0, i], V[0, i], "o", ms=3.5, mfc="white", mec="0.0", mew=0.8, zorder=3)
    ax.set_xlim(*ylim)
    ax.set_ylim(V[m][:, picks[1:4]].min() - 0.15, V[m][:, picks[1:4]].max() + 0.15)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(rf"${vs}$")
    cb = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"$t$"); cb.set_ticks([0, 5, 10, 15]); cb.ax.tick_params(direction="in")
    panel_label(ax, "b", dx=-0.17, dy=1.02)

    # (c) snapshot marginals: training times (data) and extrapolation times (evaluation sample)
    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(min(ylim[0], X.min()), max(ylim[1], X.max()), 400)
    dens = [gaussian_kde(snap[k])(xs) for k in range(16)]
    dens_ex = [gaussian_kde(X[idx(t)])(xs) for t in TEX]
    scale = 2.8 / max(d.max() for d in dens)          # tallest training density = 2.8 rows; no clipping
    hex_ = [scale * d.max() for d in dens_ex]
    rows_ex = [17.2]
    for h in hex_[:-1]:
        rows_ex.append(rows_ex[-1] + max(1.3, h + 0.15))
    for k in range(15, -1, -1):
        top = k + scale * dens[k]
        ax.fill_between(xs, k, top, color=VIR(nt(TK[k])), alpha=0.9, lw=0, zorder=-2 - 0.01 * (15 - k))
        ax.plot(xs, top, color="0.0", ls="-", lw=0.35, zorder=-1)
    for r, d in zip(rows_ex, dens_ex):
        ax.plot(xs, r + scale * d, color="0.0", lw=0.9, ls="--")
        ax.plot(xs, r + 0 * xs, color="0.0", ls="-", lw=0.4)
    ax.axhline(16.75, color="0.5", lw=0.6, ls=":")
    ax.set_yticks([0, 5, 10, 15] + rows_ex)
    ax.set_yticklabels([r"$0$", r"$5$", r"$10$", r"$15$", r"$1.5T$", r"$2T$", r"$3T$"])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_ylim(-0.4, rows_ex[-1] + hex_[-1] + 0.4); ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t_k$")
    panel_label(ax, "c", dx=-0.17, dy=1.02)

    # (d) unpaired snapshots as the model sees them
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for k in range(16):
        sub = snap[k, rng.choice(2000, 300, replace=False)]
        ax.scatter(TK[k] + rng.uniform(-0.28, 0.28, sub.size), sub, s=1.2, color=VIR(nt(TK[k])),
                   lw=0, alpha=0.8, zorder=-1)
    ax.set_xlim(-0.6, 15.6); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$t_k$"); ax.set_ylabel(r"$x$")
    panel_label(ax, "d", dx=-0.17, dy=1.02)

    save(fig, OUT / S["out"], close=True)
    print(f"[{key}] written {S['out']}")


for key in SYSTEMS:
    make(key)
