#!/bin/bash
#SBATCH --job-name=dg_data
#SBATCH --output=logs/dg_data_%j.out
#SBATCH --error=logs/dg_data_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Data prep (no GPU): download shards, build splits, build geocells.
# Run this FIRST, then run_all_gpu.sbatch.
# NOTE: deepguessr/data must point at scratch (see ICE_RUNBOOK step 4a).

set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"

module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"

echo "==> Download shards (4 train + 1 test) into scratch"
python deepguessr/src/download.py

echo "==> Build train/val/test subsets (100k train, 10k test)"
python deepguessr/src/build_subsets.py

echo "==> Build adaptive geocells on the larger train set"
python deepguessr/src/geocells.py

echo "Done. Data + geocells ready for the GPU job."
