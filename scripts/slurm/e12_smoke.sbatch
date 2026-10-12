#!/bin/bash
# E12 smoke: FastWAM + SOPT (expert_droid), 200 steps, then a 1-init closed-loop eval. Speed / memory check.
#SBATCH --job-name=sopt-e12s
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/wam_train.py --arm expert_droid --k 5 --run e12_smoke --steps 200 --bs ${BS:-16} --workers 8 2>&1 | grep -v Warning
echo "train done $(date -Is)"
python scripts/wam_eval.py --ckpt outputs/e12_smoke/final.pt --out outputs/e12_smoke/rollouts_smoke.json --n-init 1 2>&1 | tail -n 15
echo "end=$(date -Is)"
