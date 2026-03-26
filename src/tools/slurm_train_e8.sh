#!/bin/bash
#SBATCH --job-name="centerpoint_e8_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_centerpoint_e8_ro47020_%j.out
#SBATCH --error=outputs/slurm_centerpoint_e8_ro47020_%j.err

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

previous=$(nvidia-smi --query-accounted-apps="gpu_utilization,mem_utilization,max_memory_usage,time" --format=csv | /usr/bin/tail -n +2)
nvidia-smi

default_args=(model=centerpoint_radar_pfn2_gate_e8 exp_id=radar_pfn2_gate_e8 batch_size=4 num_workers=2 epochs=40)
train_args=("${default_args[@]}" "$@")

echo "Launching training with args:"
printf "  %s\n" "${train_args[@]}"

srun python -u src/tools/train.py "${train_args[@]}"

nvidia-smi --query-accounted-apps="gpu_utilization,mem_utilization,max_memory_usage,time" --format=csv | /usr/bin/grep -v -F "$previous"
