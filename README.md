# Lift–project SDEs for unpaired snapshot data

Code for the paper **"Learning hidden-state dynamics from unpaired snapshots with lift–project SDEs"** (Qi Wang, Uduak George, Gustaaf Jacobs; submitted to L4DC 2027).

Snapshot data give the distribution of an observed coordinate $x$ at a few times, but never follow an individual, so they leave the temporal coupling between times open.
The lift–project SDE lifts each observation to $z=(x,h)$ with a few hidden coordinates $h$, evolves $z$ with one autonomous SDE, and projects back to $x$; it is trained only by matching the observed marginals with a pathwise $W_2$ loss.
This repository contains the experiments and the scripts that draw every figure in the paper.

## Layout

| Folder | Paper section | Contents |
|---|---|---|
| `duffing/` | Experiments (Duffing) | Stochastic Duffing oscillator, position observed: data, training of all model classes, evaluation, processes A and B |
| `pendulum/` | Experiments (pendulum) | Damped pendulum, angle observed |
| `process_c/` | Appendix: training on marginal-equivalent data | Process C (same snapshots as A, different time correlation): parameter selection, data, evaluation |
| `shared/` | A second experiment | Shared versus experiment-conditioned dynamics under two initial distributions |
| `monotone_stat/` | Lagged covariance at comparable marginal fit | Lagged Spearman rank correlation of the stored models; Euler–Maruyama monotonicity check |
| `fucci/` | Single-cell case study | FUCCI cell-cycle data (GSE121265, GSE146773): preprocessing, training, evaluation, static readout |
| `paper_figs/` | All figures | One script per figure; `pf_common.py` holds shared loaders and style |
| `plotstyle.py` | — | Figure style (LaTeX text, no grid, PNG + EPS output) |

Model names in the code differ from the paper:

| Paper | Code |
|---|---|
| lift–project SDE ($d_h$ hidden coordinates) | `mmn` with `DH` = $d_h$ |
| obs-TH (no hidden state, time-homogeneous) | `mmn` with `DH=0` |
| obs-TD (time-dependent SDE in $x$) | `td` |
| Schrödinger bridge | `sb` |
| OT-CFM | `cfm` |
| process A / B / C | world `A` / `B` / `C` |

The `results.json`, `gate.json`, `summary_*.json` and `runs/*/{train,eval}.json` files are the stored outputs behind the numbers in the paper.
Data arrays, checkpoints and the re-simulated covariance stores (`*.npy`, `*.npz`, `*.pt`) are not included; the steps below regenerate them.

## Environment

Python 3.11 on CPU. Install with

```
pip install -r requirements.txt
```

Figures use LaTeX for text (`text.usetex=True`), so a TeX installation is needed for `paper_figs/`.

## Reproducing the results

All commands are run from the folder named in each step. Training one model takes minutes on one CPU core; the full set of runs was spread over a Slurm cluster (`*.sbatch` files) or run in parallel with `run_jobs.py`.

**Duffing** (`duffing/`)
```
python prep.py                              # training snapshots, Schrödinger-bridge pairs, evaluation truth
python train.py WORLD EXPS METHOD SEED [DH] # e.g. python train.py A 0 mmn 0 2
python run_jobs.py jobs_main.txt 4          # or run every line of the job files in parallel
python evaluate_run.py TAG                  # e.g. A0_mmn2_s0, for every trained run
python aggregate.py                         # -> results.json
python gate.py                              # pre-specified success criteria -> verdict.json
```

**Pendulum** (`pendulum/`)
```
python evaluate.py truth                    # evaluation truth and process B
sh run_all.sh jobs.txt 6                    # train.py METHOD SEED [DH] for every line
python evaluate.py run RUNNAME              # for every run
python evaluate.py aggregate                # -> results.json, eval_store.npz
python gate.py                              # -> gate.json
```

**Process C** (`process_c/`, after the Duffing steps)
```
python select_C.py                          # choose (tau, Omega) before any training on C
python prep_C.py                            # snapshots, bridge pairs and truth for process C
# train the four classes on C with the unchanged duffing/train.py (see fermi/fermi_array.sbatch)
python evaluate_C.py                        # -> results.json
```

**Shared versus conditioned dynamics** (`shared/`): `python run_jobs.py jobs.txt 4`, then `python evaluate.py TAG` for each run and `python aggregate.py`.

**Rank statistics** (`monotone_stat/`): `python monotone_stat.py duffing` and `python monotone_stat.py pendulum` after the runs are evaluated.

**FUCCI case study** (`fucci/`): download the GEO files listed in [`DATA.md`](DATA.md) into `fucci/data/raw/`, then
```
python prep_real.py                         # observed PCs, coarse bins, and the withheld phase
python run_jobs.py jobs.txt                 # train.py DS METHOD SEED [DH] (DS = hsiao, mahd)
python evaluate.py hsiao && python evaluate.py mahd
python gate.py hsiao mahd                   # -> summary_*.json, gate.json
python static_readout/static_readout.py     # supervised static readout
```
Training code opens only the observed files (`*_obs.npz`); the withheld phase is read only by `evaluate.py` and the static readout.

**Figures** (`paper_figs/`): `python resim_twotime.py duffing` and `python resim_twotime.py pendulum` re-simulate the stored models on a fine grid; then each `fig*.py` writes `out/<name>.png` and `out/<name>.eps`.

## Citation

```
@inproceedings{wang2027liftproject,
  title     = {Learning hidden-state dynamics from unpaired snapshots with lift--project {SDE}s},
  author    = {Wang, Qi and George, Uduak and Jacobs, Gustaaf},
  booktitle = {Learning for Dynamics and Control Conference (L4DC), under review},
  year      = {2027}
}
```

## License

MIT; see [`LICENSE`](LICENSE).
