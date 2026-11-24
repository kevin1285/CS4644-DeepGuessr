#!/bin/bash
#SBATCH --job-name=dg_heads_cpu
#SBATCH --output=logs/dg_heads_cpu_%j.out
#SBATCH --error=logs/dg_heads_cpu_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Baselines + geocell heads on CPU (cached embeddings -> fast). Dodges GPU queue.
set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"
module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"
export DG_DEVICE=cpu   # force CPU (handled by common.get_device override below)

python deepguessr/src/sanity_baselines.py
python deepguessr/src/baseline_zeroshot.py
python deepguessr/src/baseline_regression.py
python deepguessr/src/train_geocell_head.py --arch linear --labels onehot
python deepguessr/src/train_geocell_head.py --arch linear --labels smooth --tau 75
python deepguessr/src/train_geocell_head.py --arch mlp --labels onehot
python deepguessr/src/train_geocell_head.py --arch mlp --labels smooth --tau 75
echo "Done (CPU). All non-LoRA results at K=2533 / 190k / L-14."
