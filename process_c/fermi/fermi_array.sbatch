#!/bin/bash
# Fermi Slurm array: one training run per non-comment line of jobs.txt (world C + linear-Gaussian overlay, L4DC).
# Uses the UNCHANGED duffing train.py / common.py.  Submit from process_c/ after copying
# ../duffing/train.py and ../duffing/common.py into it (prep_C.py has written data/ there):
#   sbatch --array=1 fermi_array.sbatch      (test task)   then   sbatch --array=2-24 fermi_array.sbatch
#SBATCH -J l4dc_process_c
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=3G
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm_%A_%a.out

source /cm/shared/apps/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate torch_AE
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SLURM_SUBMIT_DIR"
args=$(grep -v '^#' jobs.txt | sed '/^\s*$/d' | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "task ${SLURM_ARRAY_TASK_ID}: $args on $(hostname)"
tag=$(echo $args | awk '{m=$3; if(m=="mmn") m="mmn"$5; print $1$2"_"m"_s"$4}')
python train.py $args > logs/${tag}.log 2>&1
echo "exit $? $(date)"
