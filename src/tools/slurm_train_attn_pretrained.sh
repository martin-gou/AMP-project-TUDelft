#!/bin/bash
#SBATCH --job-name="cp_cam_temp_attn_pt_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_cam_temp_attn_pt_%j.out
#SBATCH --error=outputs/slurm_cam_temp_attn_pt_%j.err

set -euo pipefail

mkdir -p outputs

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp
export WANDB_MODE=offline
export TORCH_HOME="$PWD/outputs/torch_home"
mkdir -p "$TORCH_HOME"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Started at: $(date)"
nvidia-smi

srun python -u src/tools/train.py \
  model=centerpoint_radar_camera_temporal_attn_pretrained \
  exp_id=centerpoint_radar_camera_temporal_attn_pretrained_slurm \
  batch_size=3 \
  num_workers=2 \
  epochs=80

echo "Finished at: $(date)"
nvidia-smi
