"""Evaluation on FRESH simulator data.
usage:
  python evaluate.py truth            -> truth.npz (ground truth, world B, noise floor, probe refs)
  python evaluate.py run RUNNAME      -> runs/RUNNAME/eval.json + eval.npz
  python evaluate.py control DH       -> controls/untrained_dhDH.json  (untrained-model probe control)
  python evaluate.py aggregate        -> results.json, eval_store.npz
Metrics (see the pre-specified protocol):
  M0      marginal W2 at the K-1 training snapshot times (fresh truth samples)
  M0_held marginal W2 at the held-out mid-times t_k+0.5 inside [0,T]
  M1      mean W2 at t in {1.5T, 2T, 3T}
  M2_in   rel. Frobenius error of two-time covariance C(t,t+d), t+d<=T ; M2_out: T<t+d<=2T
  M3      mean W2 after intervention theta -> theta+delta at T/2, at T/2+{0.5,1,2,4,7.5}
  M4      R^2(omega | theta, h) linear + MLP probes, cross-seed CCA, untrained control
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from common import CFG, TK, ROOT, build, simulate, w2_1d, save_json

torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))
T, dt, delta = CFG["T"], CFG["dt_model"], CFG["delta"]
NE = CFG["N_eval"]
idx = lambda t: int(round(t / dt))
EXT_T = [1.5 * T, 2 * T, 3 * T]
INT_T = [T / 2 + s for s in (0.5, 1, 2, 4, 7.5)]
HELD_T = list(TK[:-1] + 0.5)
C_T = np.arange(0, 2 * T + 1e-9, 1.0)
C_D = [0.5, 1, 1.5, 2, 3, 4, 6, 8]
TT, DD = np.meshgrid(C_T, C_D, indexing="ij")
MASK_IN = (TT + DD <= T + 1e-9)
MASK_OUT = (TT + DD > T + 1e-9) & (TT + DD <= 2 * T + 1e-9)
PROBE_T = [2.0, 5.0, 8.0, 11.0]
NP = 2000
WIN = 10
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
TRUTH = ROOT / "truth.npz"


def cov2(X):
    C = np.full((len(C_T), len(C_D)), np.nan)
    for i, t in enumerate(C_T):
        for j, d in enumerate(C_D):
            if t + d <= 2 * T + 1e-9:
                a, b = X[idx(t)], X[idx(t + d)]
                C[i, j] = np.mean(a * b) - a.mean() * b.mean()
    return C


def relfro(C, Cref, mask):
    return float(np.linalg.norm((C - Cref)[mask]) / np.linalg.norm(Cref[mask]))


def make_truth():
    tg, Xt, Vt = simulate(NE, 3 * T, CFG["eval_seed"], save_dt=dt)
    _, Xt2, _ = simulate(NE, 3 * T, CFG["eval_seed"] + 1, save_dt=dt)
    _, Xti, _ = simulate(NE, T, CFG["eval_seed"], save_dt=dt, intervene=(T / 2, delta))
    _, Xti2, _ = simulate(NE, T, CFG["eval_seed"] + 1, save_dt=dt, intervene=(T / 2, delta))
    U = np.random.default_rng(5).random(NE)
    XB = np.stack([np.quantile(Xt[i], U) for i in range(Xt.shape[0])])   # world B: rank-preserving
    np.savez_compressed(TRUTH, tg=tg, Xt=Xt, Vt=Vt, Xt2=Xt2, Xti=Xti, Xti2=Xti2, XB=XB)


def truth():
    return dict(np.load(TRUTH))


def metrics(X, Xi, Tr, C_true, C_B):
    Xt, Xti = Tr["Xt"], Tr["Xti"]
    m = {}
    m["M0_per_t"] = [w2_1d(X[idx(t)], Xt[idx(t)]) for t in TK[1:]]
    m["M0"] = float(np.mean(m["M0_per_t"]))
    m["M0_held_per_t"] = [w2_1d(X[idx(t)], Xt[idx(t)]) for t in HELD_T]
    m["M0_held"] = float(np.mean(m["M0_held_per_t"]))
    m["M1_per_t"] = [w2_1d(X[idx(t)], Xt[idx(t)]) for t in EXT_T]
    m["M1"] = float(np.mean(m["M1_per_t"]))
    C = cov2(X)
    m["M2_in"] = relfro(C, C_true, MASK_IN)
    m["M2_out"] = relfro(C, C_true, MASK_OUT)
    m["M2_in_vs_worldB"] = relfro(C, C_B, MASK_IN)
    m["M2_out_vs_worldB"] = relfro(C, C_B, MASK_OUT)
    m["selects"] = "A" if m["M2_in"] < m["M2_in_vs_worldB"] else "B"
    m["M3_per_t"] = [w2_1d(Xi[idx(t)], Xti[idx(t)]) for t in INT_T]
    m["M3"] = float(np.mean(m["M3_per_t"]))
    m["nonfinite"] = bool(not np.isfinite(X).all())
    m["frac_abs_theta_gt_pi_at_3T"] = float(np.mean(~(np.abs(X[-1]) <= np.pi)))
    return m, C


# ------------------------------------------------------- latent probes ---
def infer_h(model, Tr, seed=0):
    """Per-trajectory h at each probe time: fit the noise-free drift rollout over the
    next WIN steps to the observed theta window (identical for trained and untrained models)."""
    out = []
    for tp in PROBE_T:
        i0 = idx(tp)
        xobs = torch.tensor(Tr["Xt"][i0:i0 + WIN + 1, :NP], dtype=torch.float32)
        torch.manual_seed(seed)
        h = torch.zeros(NP, model.dh, requires_grad=True)
        opt = torch.optim.Adam([h], lr=0.05)
        for _ in range(300):
            z = torch.cat([xobs[0][:, None], h], 1)
            err = 0.0
            for s in range(1, WIN + 1):
                z = model.step(z, 0.0, dt, noise=False)
                err = err + ((z[:, 0] - xobs[s]) ** 2).sum()
            opt.zero_grad(); err.backward(); opt.step()
        out.append(h.detach().numpy().copy())
    return np.stack(out)


def probe_feats(Tr):
    xs = np.concatenate([Tr["Xt"][idx(t), :NP] for t in PROBE_T])
    vs = np.concatenate([Tr["Vt"][idx(t), :NP] for t in PROBE_T])
    win = np.concatenate([Tr["Xt"][idx(t):idx(t) + WIN + 1, :NP].T for t in PROBE_T])
    return xs, vs, win


def r2(F, vs):
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vs)); tr, te = perm[: len(vs) // 2], perm[len(vs) // 2:]
    lin = LinearRegression().fit(F[tr], vs[tr])
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, random_state=0).fit(F[tr], vs[tr])
    return float(lin.score(F[te], vs[te])), float(mlp.score(F[te], vs[te]))


def r2_probes(H, Tr):
    xs, vs, _ = probe_feats(Tr)
    res = {}
    res["R2_lin_x"], res["R2_mlp_x"] = r2(xs[:, None], vs)
    if H is not None:
        F = np.concatenate([xs[:, None], H.reshape(-1, H.shape[-1])], 1)
        res["R2_lin_xh"], res["R2_mlp_xh"] = r2(F, vs)
    return res


def first_cca(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    def wh(M):
        U_, S_, _ = np.linalg.svd(M, full_matrices=False)
        return U_[:, S_ > 1e-8 * S_[0]]
    return float(np.linalg.svd(wh(A).T @ wh(B), compute_uv=False)[0])


# ----------------------------------------------------------- run models ---
def load(run):
    info = json.loads((run / "train.json").read_text())
    model = build(info["method"], info["dh"], width=info["width"])
    model.load_state_dict(torch.load(run / "model.pt")); model.eval()
    return info, model


def rollouts(info, model, Tr):
    gen = torch.Generator().manual_seed(999)
    x0 = torch.tensor(Tr["Xt"][0], dtype=torch.float32)
    with torch.no_grad():
        if info["method"] == "ref":
            xa = torch.tensor(Tr["Xt"][1], dtype=torch.float32)
            X = torch.cat([x0[None], model.rollout(x0, xa, 3 * T, dt, gen=gen, t_start=dt)])
            gen = torch.Generator().manual_seed(999)
            Xi = torch.cat([x0[None], model.rollout(x0, xa, T, dt, gen=gen, t_start=dt,
                                                     intervene=(T / 2, delta))])
        elif info["method"] == "cfm":
            X = model.rollout(x0, 3 * T, dt)
            Xi = model.rollout(x0, T, dt, intervene=(T / 2, delta))
        else:
            X = model.rollout(x0, 3 * T, dt, gen=gen)
            gen = torch.Generator().manual_seed(999)
            Xi = model.rollout(x0, T, dt, gen=gen, intervene=(T / 2, delta))
    return X.numpy(), Xi.numpy()


def eval_run(name):
    Tr = truth()
    C_true, C_B = cov2(Tr["Xt"]), cov2(Tr["XB"])
    run = ROOT / "runs" / name
    if not run.exists():   # debug evaluation of pilot runs only
        run = ROOT / "pilots" / name
    info, model = load(run)
    X, Xi = rollouts(info, model, Tr)
    m, C = metrics(X, Xi, Tr, C_true, C_B)
    m.update(method=info["method"], seed=info["seed"], dh=info["dh"], nparams=info["nparams"],
             width=info["width"], train_iters=info["iters"], train_wall_s=info["wall_s"],
             final_train_loss=info["hist"][-1][2])
    store = dict(q=np.quantile(X, QS, axis=1), qi=np.quantile(Xi, QS, axis=1), C=C, mean=X.mean(1),
                 meani=Xi.mean(1))
    out = {name: m}
    if info["method"] in ("td", "tdw", "cfm"):
        model.tclip = T   # second continuation: time input frozen at T beyond T
        Xc, _ = rollouts(info, model, Tr)
        mc, Cc = metrics(Xc, Xi, Tr, C_true, C_B)
        mc.update(method=info["method"] + "clip", seed=info["seed"], dh=0, nparams=info["nparams"])
        out[name.replace("_s", "clip_s")] = mc
        store.update(clip_mean=Xc.mean(1), clip_q=np.quantile(Xc, QS, axis=1), clip_C=Cc)
    if info["method"] == "mmn":
        H = infer_h(model, Tr) if info["dh"] > 0 else None
        m.update(r2_probes(H, Tr))
        if H is not None:
            store["H"] = H
    save_json(out, run / "eval.json")
    np.savez_compressed(run / "eval.npz", **store)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                 if isinstance(v, (float, str)) and k != "method"}, flush=True)


def control(dh):
    Tr = truth()
    res = {}
    for s in range(CFG["seeds"]):
        torch.manual_seed(1000 + s)
        ctrl = build("mmn", dh)
        res[f"s{s}"] = r2_probes(infer_h(ctrl, Tr), Tr)
    (ROOT / "controls").mkdir(exist_ok=True)
    save_json(res, ROOT / "controls" / f"untrained_dh{dh}.json")
    print("control", dh, res)


def aggregate():
    Tr = truth()
    Xt = Tr["Xt"]
    C_true, C_B = cov2(Xt), cov2(Tr["XB"])
    nf, _ = metrics(Tr["Xt2"], Tr["Xti2"], Tr, C_true, C_B)
    worldB = dict(M2_in_truthB_vs_truthA=relfro(C_B, C_true, MASK_IN),
                  M2_out_truthB_vs_truthA=relfro(C_B, C_true, MASK_OUT),
                  max_marginal_W2_A_vs_B=float(max(w2_1d(Tr["XB"][i], Xt[i]) for i in range(0, Xt.shape[0], 10))))
    xs, vs, win = probe_feats(Tr)
    probe_ref = {}
    probe_ref["R2_lin_window"], probe_ref["R2_mlp_window"] = r2(win, vs)
    results = dict(cfg=CFG, noise_floor=nf, worldB=worldB, probe_reference_theta_window=probe_ref,
                   truth_max_abs_theta=float(np.abs(Xt).max()),
                   truth_max_abs_theta_intervened=float(np.abs(Tr["Xti"]).max()), runs={})
    store = dict(tg=Tr["tg"], q_true=np.quantile(Xt, QS, axis=1), q_true_int=np.quantile(Tr["Xti"], QS, axis=1),
                 mean_true=Xt.mean(1), C_true=C_true, C_B=C_B, C_T=C_T, C_D=np.array(C_D),
                 v_probe=vs, x_probe=xs)
    H_by = {}
    for run in sorted((ROOT / "runs").iterdir()):
        if not (run / "eval.json").exists():
            continue
        results["runs"].update(json.loads((run / "eval.json").read_text()))
        ev = np.load(run / "eval.npz")
        for k in ev.files:
            if k == "H":
                H_by[run.name] = ev["H"]
            elif k.startswith("clip_"):
                store[f"{k[5:]}_{run.name.replace('_s', 'clip_s')}"] = ev[k]
            else:
                store[f"{k}_{run.name}"] = ev[k]
    # cross-seed CCA of inferred latents
    cons = {}
    for dh in (1, 2, 4):
        names = [n for n in sorted(H_by) if n.startswith(f"mmn_dh{dh}_")]
        vals = [float(np.mean([first_cca(H_by[a][p], H_by[b][p]) for p in range(len(PROBE_T))]))
                for a, b in itertools.combinations(names, 2)]
        cons[f"dh{dh}"] = dict(pairs=vals, mean=float(np.mean(vals)) if vals else None,
                               median=float(np.median(vals)) if vals else None)
    results["latent_consistency_cca"] = cons
    # correlation of h with omega (dh=1) per seed; and first-CCA(h, omega) for every dh
    hv = {}
    for n, H in H_by.items():
        Hp = H.reshape(-1, H.shape[-1])
        hv[n] = dict(cca_h_omega=first_cca(Hp, vs[:, None]),
                     rho_h_omega=float(np.corrcoef(Hp[:, 0], vs)[0, 1]) if H.shape[-1] == 1 else None)
        store[f"H_{n}"] = H[1]
    results["h_vs_omega"] = hv
    ctr = {}
    for dh in (1, 2, 4):
        f = ROOT / "controls" / f"untrained_dh{dh}.json"
        if f.exists():
            ctr[f"dh{dh}"] = json.loads(f.read_text())
    results["probe_control_untrained"] = ctr
    store["v_probe_t5"] = Tr["Vt"][idx(PROBE_T[1]), :NP]
    np.savez_compressed(ROOT / "eval_store.npz", **store)
    save_json(results, ROOT / "results.json")
    print("aggregate done:", len(results["runs"]), "entries")


if __name__ == "__main__":
    t0 = time.time()
    cmd = sys.argv[1]
    if cmd == "truth":
        make_truth()
    elif cmd == "run":
        eval_run(sys.argv[2])
    elif cmd == "control":
        control(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
    print(cmd, "wall", f"{time.time()-t0:.0f}s")
