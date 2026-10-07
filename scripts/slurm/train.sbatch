#!/bin/bash
# Jubail. Keep at most 2 SOPT jobs queued/running at once.
#   sbatch --job-name=e2_b_f010 scripts/slurm/train.sbatch --config configs/model/b.yaml data.train_fraction=0.1 run_name=e2_b_f010
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /scratch/ll5582/SOPT
source scripts/remote/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py data.root=/scratch/ll5582/SOPT_DATA/processed/droid train.num_workers=14 "$@"
