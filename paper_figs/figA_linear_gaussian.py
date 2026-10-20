"""Appendix figure: exact linear-Gaussian example (same marginals, different two-time covariances, no fitting).

System: partially observed underdamped oscillator dx = v dt, dv = (-0.2 v - x) dt + 0.15 dW (Duffing with beta = 0),
(x0, v0) ~ N((1.5, 0), 0.2^2 I); only x is observed.
(a) exact marginals m_x(t) +- s_t (shared by every coupling below);
(b) C(t0, t0+Delta) at t0 = 2: true system (= lifted realization with d_h = 1), comonotone coupling, Markov mimic in x
    (computed from its own drift integral; coincides with the comonotone curve), best Gauss-Markov x-diffusion;
(c) eps_C to the true covariance for the Gauss-Markov x-diffusion family versus its diffusion level c, with the
    lower bound for every x-Markov diffusion (Proposition in the paper) and the independent coupling (eps_C = 1).
The optional neural overlay of linear_gaussian/ (2 seeds) is not plotted.
Sources: linear_gaussian/results.json (exact.epsC: markov_best_c, markov_best_vs_A, B_comonotone_vs_A,
lower_bound_any_xMarkov_diffusion_vs_A) and the exact formulas in linear_gaussian/lg_exact.py (imported, no sampling).
usage: python figA_linear_gaussian.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from pf_common import COLW, LG, LG_R, OUT, panel_label, save, style

sys.path.insert(0, str(LG))
import lg_exact as L  # noqa: E402

style()
ec = LG_R["exact"]["epsC"]
t0 = 2.0
dd = np.linspace(0, 13, 261)
dg = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8])
cstar = ec["markov_best_c"]

fig, ax = plt.subplots(1, 3, figsize=(COLW, 2.05), gridspec_kw=dict(wspace=0.62))
fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.36)

# (a) exact marginals
tt = np.linspace(0, 15, 301)
mo = [L.moments(t) for t in tt]
mx = np.array([m[0][0] for m in mo]); sx = np.sqrt([m[1][0, 0] for m in mo])
a = ax[0]
a.fill_between(tt, mx - 2 * sx, mx + 2 * sx, color="0.82", lw=0)
a.plot(tt, mx, "k-", lw=1.4)
a.set_xlabel(r"$t$"); a.set_ylabel(r"$x$"); a.set_xlim(0, 15)
panel_label(a, "a", dx=-0.30, dy=1.04)

# (b) two-time covariance against lag
a = ax[1]
a.axhline(0, color="0.8", lw=0.5, zorder=-5)
a.plot(dd, [L.C_true(t0, t0 + d) for d in dd], "k-", lw=1.8, label=r"true $=$ lift--project, $d_h{=}1$")
a.plot(dd, [L.C_como(t0, t0 + d) for d in dd], color="0.5", ls="--", lw=1.5, label=r"comonotone")
a.plot(dg, [L.C_gyongy(t0, t0 + d) for d in dg], ls="none", marker="o", mfc="white", mec="0.15", ms=4,
       label=r"Markov mimic in $x$")
a.plot(dd, [L.C_markov(cstar)(t0, t0 + d) for d in dd], color="0.2", ls="-.", lw=1.1,
       label=rf"best constant-$c$ ($c^\star{{=}}{cstar:.2f}$)")
a.set_xlabel(r"$d$"); a.set_ylabel(r"$C(2,2{+}d)$"); a.set_xlim(0, 13)
panel_label(a, "b", dx=-0.30, dy=1.04)
fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7.0, handlelength=2.2,
           columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))

# (c) Gauss-Markov family and the bound
a = ax[2]
cs = np.linspace(0, 1.2, 61)
CAg = L.grid(L.C_true)
a.plot(cs, [L.eps(L.grid(L.C_markov(x)), CAg) for x in cs], "k-", lw=1.4)
lb = ec["lower_bound_any_xMarkov_diffusion_vs_A"]
a.axhline(lb, color="0.45", ls="--", lw=1.1)
a.text(1.18, lb - 0.06, r"bound", ha="right", va="top", fontsize=7)
a.axhline(1.0, color="0.65", ls=":", lw=1.0)
a.text(1.18, 1.03, r"independent", ha="right", va="bottom", fontsize=7)
a.plot([0], [ec["B_comonotone_vs_A"]], ls="none", marker="o", mfc="white", mec="k", ms=4.5, clip_on=False)
a.plot([cstar], [ec["markov_best_vs_A"]], ls="none", marker="D", mfc="k", mec="k", ms=3.5)
a.set_xlabel(r"$c$"); a.set_ylabel(r"$\epsilon_C$"); a.set_ylim(0, 2.1); a.set_xlim(0, 1.2)
panel_label(a, "c", dx=-0.30, dy=1.04)
print("eps comonotone", round(ec["B_comonotone_vs_A"], 3), "best GM", round(ec["markov_best_vs_A"], 3), "c*", round(cstar, 3),
      "bound", round(lb, 3), "frac neg", round(ec["frac_negative_true_cov_entries"], 3),
      "duffing bound", round(LG_R["exact"]["duffing_lower_bound_any_xMarkov_diffusion_vs_A"], 3))
save(fig, OUT / "figA_linear_gaussian", close=True)
print("figA written")
