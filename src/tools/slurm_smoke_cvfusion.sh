#!/bin/bash
#SBATCH --job-name="cvfusion_smoke_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=0:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_cvfusion_smoke_%j.out
#SBATCH --error=outputs/slurm_cvfusion_smoke_%j.err

set -euo pipefail

mkdir -p outputs

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

export WANDB_MODE=offline

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Started at: $(date)"
nvidia-smi

srun python -u src/tools/train.py \
  model=cvfusion \
  exp_id=cvfusion_smoke \
  batch_size=1 \
  num_workers=0 \
  epochs=1 \
  sync_bn=false \
  fast_dev_run=true \
  limit_train_batches=1 \
  limit_val_batches=1 \
  log_every=1

echo "Finished at: $(date)"
nvidia-smi
