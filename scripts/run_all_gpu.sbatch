#!/bin/bash
#SBATCH --job-name=dg_all
#SBATCH --output=logs/dg_all_%j.out
#SBATCH --error=logs/dg_all_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Full model suite at final scale (GPU). Run AFTER prep_data.sbatch.
# Reproduces every table row on the larger data, then trains LoRA.

set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"

echo "==> [1/6] Frozen CLIP embeddings (resumable)"
python deepguessr/src/extract_embeddings.py

echo "==> [2/6] Sanity baselines"
python deepguessr/src/sanity_baselines.py

echo "==> [3/6] Zero-shot CLIP (single + ensemble)"
python deepguessr/src/baseline_zeroshot.py

echo "==> [4/6] Regression (linear + MLP)"
python deepguessr/src/baseline_regression.py

echo "==> [5/6] Geocell heads (linear/mlp x onehot/smooth)"
python deepguessr/src/train_geocell_head.py --arch linear --labels onehot
python deepguessr/src/train_geocell_head.py --arch linear --labels smooth --tau 75
python deepguessr/src/train_geocell_head.py --arch mlp --labels onehot
python deepguessr/src/train_geocell_head.py --arch mlp --labels smooth --tau 75

echo "==> [6/6] LoRA fine-tuning (warm-started from frozen MLP smoothed head)"
python deepguessr/src/train_lora.py --labels smooth --tau 75

echo "Done. All results in deepguessr/results/."
