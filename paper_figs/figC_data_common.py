"""Shared helpers for the colour data-illustration figures of the L4DC appendix (figC_*.py).

Colour is used on purpose in these figures (author request): a continuous quantity is mapped to a
perceptually uniform colormap (viridis), a phase to a cyclic colormap (twilight), and a signed
covariance to a diverging colormap centred at zero (RdBu_r).  Everything else follows the house style
(pf_common.style: LaTeX text, no grid, frameless legends, panel labels outside the axes, PNG + EPS).
"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# viridis without its lightest yellow, so thin lines stay visible on white paper
VIR = LinearSegmentedColormap.from_list("viridis_print", mpl.colormaps["viridis"](np.linspace(0.0, 0.88, 256)))


def colored_line(ax, x, y, c, cmap, norm, lw=1.1, zorder=-1):
    """Draw the polyline (x, y) with segment colour given by c (length len(x))."""
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder, capstyle="round")
    lc.set_array(0.5 * (c[:-1] + c[1:]))
    ax.add_collection(lc)
    return lc


def rank_picks(x0, qs=(0.08, 0.3, 0.5, 0.7, 0.92)):
    """Indices of the particles whose initial value is closest to the given quantiles of x0."""
    return [int(np.argmin(np.abs(x0 - np.quantile(x0, q)))) for q in qs], list(qs)


def two_time_cov(X, dt, tmax, dmax):
    """C(t, t+d) = Cov(x_t, x_{t+d}) on the sampling grid, t in [0, tmax], d in [0, dmax].

    X has shape (n_times, n_particles) with time step dt.  Returns t, d, C (shape len(t) x len(d)).
    """
    nt, nd = int(round(tmax / dt)) + 1, int(round(dmax / dt)) + 1
    Xc = X[: nt + nd - 1] - X[: nt + nd - 1].mean(1, keepdims=True)
    C = np.empty((nt, nd))
    for j in range(nd):
        C[:, j] = np.mean(Xc[:nt] * Xc[j: j + nt], axis=1)
    return np.arange(nt) * dt, np.arange(nd) * dt, C
