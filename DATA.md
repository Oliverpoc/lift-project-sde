# Data

The Duffing, pendulum, process-C and linear-Gaussian data are simulated by the `prep*.py` scripts and need no download.

The FUCCI case study uses two public single-cell data sets from NCBI GEO. Place the files below in `fucci/data/raw/` and run `fucci/prep_real.py`.

## GSE121265 (iPSC, 888 cells)

Hsiao et al. (2020), "Characterizing and inferring quantitative cell cycle phase in single-cell RNA-seq data analysis", *Genome Research*.

- Source: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121265/suppl/
- File used: `GSE121265_eset-final.rds.gz`, gunzipped to `.rds`. It is the final ExpressionSet (also at github.com/jhsiao999/fucci-seq) and carries the FUCCI intensities. It is read in Python with the `rdata` package.
- Fields used: UMI counts; the individual (`chip_id`) for batch centring; the FUCCI angle `theta` and the adjusted GFP/RFP intensities, which set the orientation and origin of the withheld phase only.
- The FUCCI transgenes (EGFP, mCherry) and ERCC spike-ins are removed before any feature is computed.

## GSE146773 (U2OS, 1146 cells)

Mahdessian et al. (2021), "Spatiotemporal dissection of the cell cycle with single-cell proteogenomics", *Nature*.

- Source: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE146nnn/GSE146773/suppl/
- Files used: `GSE146773_Counts.csv.gz` and `GSE146773_fucci_coords.csv.gz`.
- Fields used: counts; the plate (from the cell id) for batch centring; `fucci_time_hrs` for the withheld phase.
- Cells used: the 1146 cells that have FUCCI coordinates.

`prep_real.py` writes, for each data set, an observed file (`*_obs.npz`: six principal components and the coarse phase bin) and a withheld file (`*_truth_WITHHELD.npz`: the continuous phase). Training code refuses to open the withheld file.
