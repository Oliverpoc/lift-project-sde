"""Shared loaders and style for the L4DC 2027 paper figures.

Every number plotted here is read from stored results of the pre-registered runs:
  duffing/results.json, duffing/data/{truth.npz,distinguishability.json},
  duffing/runs/<tag>/store.npz,
  pendulum/gate.json (summary block), pendulum/eval_store.npz,
  worldC/results.json (process-C test), linear_gaussian/results.json (exact Gaussian example).
Nothing is re-trained or re-evaluated. Figures are written as PNG + EPS (never PDF)
to paper_figs/out/.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4DC = HERE.parent
ROOT = L4DC
OUT = HERE / "out"
sys.path.insert(0, str(ROOT))                       # project-root plotstyle.py
from plotstyle import use_style, panel_label, save  # noqa: E402,F401

# PMLR (jmlr class, pmlr option) text width measured from the template: 433.62 pt = 6.0 in.
COLW = 6.0
BASE = 9.0          # base font size (pt) at print size; figures are included at width=\textwidth


def style():
    use_style(base=BASE)
    import matplotlib as mpl
    mpl.rcParams.update({"lines.linewidth": 1.4, "lines.markersize": 4.5, "lines.markeredgewidth": 0.8,
                         "legend.handlelength": 2.2, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
                         "xtick.minor.size": 2.0, "ytick.minor.size": 2.0, "axes.linewidth": 0.8})


# ----------------------------------------------------------------------------- Duffing
DUF = L4DC / "duffing"
DUF_R = json.loads((DUF / "results.json").read_text())
DUF_G = DUF_R["groups"]
DUF_Z = np.load(DUF / "data" / "truth.npz")
DUF_DIST = json.loads((DUF / "data" / "distinguishability.json").read_text())
DUF_T, DUF_DT = 15.0, 0.1
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]          # lags of the two-time covariance (pre-specified protocol)
C_T = np.arange(0, 2 * DUF_T + 1e-9, 1.0)


def duf(group, metric):
    """Per-seed values of a Duffing metric (results.json -> groups)."""
    g = DUF_G.get(group, {})
    return np.array(g[metric]["per_seed"]) if metric in g else np.array([])


def duf_cov(X):
    """Two-time covariance C(t, t+d) on the C_T x C_D grid (same as duffing/figures.py)."""
    idx = lambda t: int(round(t / DUF_DT))
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if idx(t + d) < X.shape[0]:
                a, b = X[idx(t)], X[idx(t + d)]
                C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def duf_store_median(group, key="C0"):
    arrs = []
    for s in range(5):
        p = DUF / "runs" / f"{group}_s{s}" / "store.npz"
        if p.exists():
            st = np.load(p)
            if key in st.files:
                arrs.append(st[key])
    return np.nanmedian(arrs, 0) if arrs else None


# ----------------------------------------------------------------------------- pendulum
PEN = L4DC / "pendulum"
PEN_S = json.loads((PEN / "gate.json").read_text())["summary"]
PEN_ST = np.load(PEN / "eval_store.npz")
PEN_T = 15.0


def pen(cls, metric):
    """Per-seed values of a pendulum metric (gate.json -> summary)."""
    return np.array(PEN_S[cls][metric]["per_seed"])


def pen_store_median(pattern):
    arrs = [PEN_ST[pattern.format(s=s)] for s in range(5) if pattern.format(s=s) in PEN_ST.files]
    return np.median(arrs, 0)


# ----------------------------------------------------------------------------- process C (third coupling)
WC = L4DC / "worldC"
WC_R = json.loads((WC / "results.json").read_text())


def wc(group, metric):
    """Per-seed eps_C of a process-C study group (worldC/results.json -> groups; A0_*/B0_* are the reused Duffing runs)."""
    return np.array(WC_R["groups"][group][metric]["per_seed"])


# ----------------------------------------------------------------------------- linear-Gaussian example
LG = L4DC / "linear_gaussian"
LG_R = json.loads((LG / "results.json").read_text())


# ----------------------------------------------------------------------------- class palette (main-text colour figures)
# Author decision (2026-10-09): the main-text figures of the L4DC paper are in colour.  One qualitative palette,
# used identically in every main figure: the lift--project SDE (our method) is a strong saturated red-orange,
# the observed-state baselines are distinct muted colours, the truth (process A) is black.  Markers stay distinct
# per class so the figures remain readable in greyscale.
PAL = {
    "lp": "#D7301F",     # lift--project SDE, d_h = 2
    "lp1": "#F28E2B",    # lift--project SDE, d_h = 1
    "th": "#8DA0CB",     # obs-TH (d_h = 0)
    "td": "#3E6FA8",     # obs-TD
    "tdw": "#72B7B2",    # obs-TD-warm
    "cfm": "#9C755F",    # OT-CFM
    "sb": "#A87CB8",     # SB
    "truth": "#000000",  # process A
    "B": "#8C8C8C",      # process B
}
MK = {"lp": "o", "lp1": "h", "th": "v", "td": "s", "tdw": "P", "cfm": "^", "sb": "D"}
LAB = {"lp": r"lift--project", "lp1": r"lift--project $d_h{=}1$", "th": r"obs-TH", "td": r"obs-TD",
       "tdw": r"obs-TD-warm", "cfm": r"OT-CFM", "sb": r"SB"}
