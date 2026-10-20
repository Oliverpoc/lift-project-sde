"""Preprocess GSE121265 (Hsiao FUCCI-seq, iPSC) and GSE146773 (Mahdessian U2OS FUCCI SMART-seq2).

Writes, per dataset DS:
  data/DS_obs.npz            observable only: X (N,d) PCA coords, bin (N,) coarse FUCCI bin, K, hold, cell ids
  data/DS_truth_WITHHELD.npz withheld truth: psi (population-quantile FUCCI phase, rad), raw FUCCI angle, intensities
Training code may open ONLY *_obs.npz (enforced by common.guard_no_truth()).

Pipeline (identical for both, no FUCCI information used in X):
  remove non-human features (FUCCI transgene EGFP/mCherry, ERCC);
  library-size normalise to 1e4, log1p; keep genes detected in >=10% of cells;
  centre each gene within technical batch (Hsiao: individual; Mahdessian: plate);
  top NHVG genes by variance; z-score; PCA to d components; divide all PCs by sd(PC1).
Truth: FUCCI phase angle, oriented so that increasing angle = cell-cycle progression, origin at the
post-mitotic low/low FUCCI point; psi = 2*pi * (circular rank from the origin)/N (population quantile).
Coarse bins: bin = floor(K * psi / 2pi), K equal-count bins.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
D_PCA, NHVG, K, HOLD, MIN_DET = 6, 1000, 6, 3, 0.10


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pca_obs(counts, batch):
    """counts: (N cells, G genes) float. Returns X (N, D_PCA) and info."""
    lib = counts.sum(1, keepdims=True)
    L = np.log1p(counts / lib * 1e4)
    keep = (counts > 0).mean(0) >= MIN_DET
    L = L[:, keep]
    for b in np.unique(batch):
        m = batch == b
        L[m] -= L[m].mean(0)
    v = L.var(0)
    hv = np.argsort(v)[::-1][:NHVG]
    Z = L[:, hv]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = U[:, :D_PCA] * S[:D_PCA]
    evr = (S ** 2 / (S ** 2).sum())[:D_PCA]
    X = P / P[:, 0].std()
    return X.astype(np.float32), dict(n_genes_detected=int(keep.sum()), n_hvg=NHVG, evr=evr.tolist())


def quantile_phase(angle, origin):
    a = np.mod(angle - origin, 2 * np.pi)
    r = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    return (r + 0.5) / len(a) * 2 * np.pi, a


def write(ds, X, ids, psi, extra_truth, info):
    b = np.floor(psi * K / (2 * np.pi)).astype(int)
    np.savez(ROOT / "data" / f"{ds}_obs.npz", X=X, bin=b, K=K, hold=HOLD, ids=np.array(ids))
    np.savez(ROOT / "data" / f"{ds}_truth_WITHHELD.npz", psi=psi, ids=np.array(ids), **extra_truth)
    info.update(N=int(len(X)), d=D_PCA, K=K, hold=HOLD, bin_counts=np.bincount(b, minlength=K).tolist())
    (ROOT / "data" / f"{ds}_prep_info.json").write_text(json.dumps(info, indent=1))
    print(ds, info)


def hsiao():
    import rdata
    f = RAW / "GSE121265_eset-final.rds"
    c = rdata.conversion.convert(rdata.parser.parse_file(f))
    p = c.phenoData.data
    E = c.assayData.maps[0]["exprs"]
    fd = c.featureData.data
    human = (fd["source"].astype(str).values == "H. sapiens")
    counts = np.asarray(E.values, dtype=np.float64)[human].T          # cells x human genes
    batch = p["chip_id"].astype(str).values
    X, info = pca_obs(counts, batch)
    th = p["theta"].astype(float).values
    r = p["rfp.median.log10sum.adjust"].astype(float).values
    g = p["gfp.median.log10sum.adjust"].astype(float).values
    # orientation checked in data exploration: increasing theta = low/low -> RFP(G1) -> both(S) -> GFP(G2/M)
    # origin = circularly smoothed minimum of rfp+gfp (post-mitotic early G1)
    grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    dd = np.angle(np.exp(1j * (grid[:, None] - th[None, :])))
    w = np.exp(-0.5 * (dd / 0.3) ** 2)
    s = (w * (r + g)[None]).sum(1) / w.sum(1)
    origin = grid[np.argmin(s)]
    psi, a = quantile_phase(th, origin)
    info.update(source="GSE121265_eset-final.rds (pData theta, rfp/gfp .median.log10sum.adjust; exprs counts)",
                sha256=sha(f), batch="chip_id (individual)", n_human_genes=int(human.sum()),
                excluded_features=fd.index[~human].tolist(), theta_origin=float(origin))
    write("hsiao", X, list(p.index.astype(str)), psi, dict(theta=th, angle_from_origin=a, rfp=r, gfp=g), info)


def mahdessian():
    fc = RAW / "GSE146773_fucci_coords.csv.gz"
    fx = RAW / "GSE146773_Counts.csv.gz"
    co = pd.read_csv(fc)
    M = pd.read_csv(fx, index_col=0)
    human = [col for col in M.columns if col.startswith("ENSG")]
    M = M.loc[co.cell.values, human]
    counts = M.values.astype(np.float64)
    batch = np.array([s.split("_")[1] for s in co.cell.values])
    X, info = pca_obs(counts, batch)
    tt = co.fucci_time_hrs.values.astype(float)
    ang = 2 * np.pi * tt / (tt.max() + 1e-9)        # pseudotime 0 = start of G1 (authors' convention)
    psi, a = quantile_phase(ang, 0.0)
    info.update(source="GSE146773_Counts.csv.gz (RSEM expected counts) + GSE146773_fucci_coords.csv.gz (fucci_time_hrs)",
                sha256_counts=sha(fx), sha256_coords=sha(fc), batch="plate (355/356/357)",
                n_human_genes=len(human), n_cells_counts_file=1152, n_cells_with_fucci=int(len(co)))
    write("mahd", X, list(co.cell.values), psi,
          dict(fucci_time_hrs=tt, polar_coord_phi=co.polar_coord_phi.values, green=co.green530_lognorm_rescale.values,
               red=co.red585_lognorm_rescale.values, facs_phase=co.phase_by_facs_gating.astype(str).values), info)


if __name__ == "__main__":
    hsiao()
    mahdessian()
