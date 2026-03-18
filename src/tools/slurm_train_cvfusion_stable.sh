#!/bin/bash
#SBATCH --job-name="cvfusion_stable_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=slurm_cvfusion_stable_%j.out
#SBATCH --error=slurm_cvfusion_stable_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p outputs

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

EXP_ID="cvfusion_3s_pretrained_lr1e3_ep80"
COMMON_OVERRIDES=(
  "model=cvfusion"
  "model.num_sweeps=3"
  "model.stage1.image_backbone.pretrained=true"
  "model.stage1.image_backbone.freeze_bn=true"
  "model.optimizer.lr=0.001"
  "batch_size=2"
  "num_workers=2"
  "epochs=80"
  "sync_bn=false"
  "val_every=3"
  "log_every=20"
)

RADAR_3_DIR=""
for candidate in \
  "data/view_of_delft/radar_3_scans/training/velodyne" \
  "data/view_of_delft/radar_3frames/training/velodyne" \
  "data/view_of_delft/radar_3_frames/training/velodyne"
do
  if [ -d "$candidate" ]; then
    RADAR_3_DIR="$candidate"
    break
  fi
done

if [ -z "$RADAR_3_DIR" ]; then
  echo "Missing 3-sweep radar data under data/view_of_delft." >&2
  echo "Expected one of: radar_3_scans, radar_3frames, radar_3_frames" >&2
  exit 1
fi

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "3-sweep radar dir: $RADAR_3_DIR"
echo "Started at: $(date)"
nvidia-smi

echo "Running CVFusion smoke test before full training..."
srun --kill-on-bad-exit=1 python -u src/tools/train.py \
  "${COMMON_OVERRIDES[@]}" \
  exp_id="${EXP_ID}_smoke" \
  batch_size=1 \
  num_workers=0 \
  epochs=1 \
  fast_dev_run=true \
  limit_train_batches=1 \
  limit_val_batches=1 \
  log_every=1

echo "Smoke test passed. Starting full training..."
srun --kill-on-bad-exit=1 python -u src/tools/train.py \
  "${COMMON_OVERRIDES[@]}" \
  exp_id="${EXP_ID}" \
  fast_dev_run=false \
  limit_train_batches=1.0 \
  limit_val_batches=1.0

echo "Finished at: $(date)"
nvidia-smi
