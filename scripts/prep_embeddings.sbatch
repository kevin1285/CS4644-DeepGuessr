#!/bin/bash
#SBATCH --job-name=dg_prep
#SBATCH --output=logs/dg_prep_%j.out
#SBATCH --error=logs/dg_prep_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Prep job: extract frozen CLIP embeddings (GPU) + train the frozen MLP smoothed
# geocell head (produces the warm-start file for LoRA + the E6 frozen result).

set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"

echo "==> Extracting frozen CLIP embeddings (resumable)"
python deepguessr/src/extract_embeddings.py

echo "==> Training frozen MLP smoothed head (saves warm-start file)"
python deepguessr/src/train_geocell_head.py --arch mlp --labels smooth --tau 75

echo "Done. Embeddings + frozen_head_mlp_smooth_tau75.pt ready."
