#!/bin/bash
#SBATCH --job-name=dg_lora
#SBATCH --output=logs/dg_lora_%j.out
#SBATCH --error=logs/dg_lora_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# LoRA only (embeddings + frozen heads already saved). Lightweight checkpoints.
set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi || true
module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"
python deepguessr/src/train_lora.py --labels smooth --tau 75
echo "Done."
