#!/bin/bash
#SBATCH --job-name=dg_lora
#SBATCH --output=logs/dg_lora_%j.out
#SBATCH --error=logs/dg_lora_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# NOTE: you may need to add a partition/account line for your ICE allocation, e.g.
#   #SBATCH --partition=ice-gpu
# Check with 'sinfo' what partitions you can use; see runbook step 5.

set -euo pipefail
mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

# --- environment ---
module load anaconda3 2>/dev/null || module load anaconda3/2023.03 || true
conda activate deepguessr

# --- run LoRA fine-tuning (E5) ---
# Defaults come from configs/default.yaml (lora: block). Override via flags.
# train-images / epochs chosen so one run is <= ~6h; tune after timing throughput.
cd "$SLURM_SUBMIT_DIR"

python deepguessr/src/train_lora.py \
    --labels smooth --tau 75 \
    --epochs 3 \
    --train-images 100000

echo "Done. Results in deepguessr/results/ (lora_geocell_*.json, *_preds.csv, *_curve.csv)"
