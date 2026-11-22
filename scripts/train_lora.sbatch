#!/bin/bash
#SBATCH --job-name=dg_lora
#SBATCH --output=logs/dg_lora_%j.out
#SBATCH --error=logs/dg_lora_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# GPU choice: l40s is fast + has 48GB (plenty for LoRA on an 86M model).
# If l40s nodes are busy and the job sits pending, swap the --gres line for a
# generic request:   #SBATCH --gres=gpu:1
# or another type seen in `sinfo`, e.g. gpu:a100:1 / gpu:v100:1.

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
    --train-images 100000

echo "Done. Results in deepguessr/results/ (lora_geocell_*.json, *_preds.csv, *_curve.csv)"
