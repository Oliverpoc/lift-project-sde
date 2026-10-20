#!/bin/bash
# Fermi Slurm array: one training run per line of jobs.txt (shared vs conditioned, L4DC).
# Submit from this folder:  sbatch --array=1-25 fermi_array.sbatch
#SBATCH -J l4dc_shared
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm_%A_%a.out

source /cm/shared/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate torch_AE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
args=$(grep -v '^#' jobs.txt | sed '/^\s*$/d' | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "task ${SLURM_ARRAY_TASK_ID}: $args on $(hostname)"
python train.py $args
