#!/bin/bash
#SBATCH --job-name=dg_heads
#SBATCH --output=logs/dg_heads_%j.out
#SBATCH --error=logs/dg_heads_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Rerun baselines + geocell heads at the CURRENT config (K=2533, 190k, L/14),
# reusing cached embeddings. LoRA already done separately. Makes every model
# match LoRA's data/cells for a fair comparison.
set -euo pipefail
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)"; nvidia-smi || true
module load anaconda3/2023.03 2>/dev/null || true
conda activate deepguessr
cd "$SLURM_SUBMIT_DIR"

# Embeddings must already be cached (L/14, 190k). extract is resumable/no-op if so.
python deepguessr/src/extract_embeddings.py
python deepguessr/src/sanity_baselines.py
python deepguessr/src/baseline_zeroshot.py
python deepguessr/src/baseline_regression.py
python deepguessr/src/train_geocell_head.py --arch linear --labels onehot
python deepguessr/src/train_geocell_head.py --arch linear --labels smooth --tau 75
python deepguessr/src/train_geocell_head.py --arch mlp --labels onehot
python deepguessr/src/train_geocell_head.py --arch mlp --labels smooth --tau 75
echo "Done. All non-LoRA results now at K=2533 / 190k / L-14."
