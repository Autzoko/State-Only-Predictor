#!/bin/bash
# E37 probe follow-up: source of reset-to-reset differences in LIBERO kitchen scenes (scripts/e37_diag.py; DIAG=e37_diag2 drift test, DIAG=e37_diag3 texture trace).
#   sbatch scripts/slurm/e37_diag.sbatch
#SBATCH --job-name=sopt-e37d
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/${DIAG:-e37_diag}.py 2>&1 | grep -v "task orders"
echo "end=$(date -Is)"
