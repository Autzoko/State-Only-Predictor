#!/bin/bash
# E7 smoke test: expert-conditioned SmolVLA trains a few steps and runs a short closed-loop episode.
#SBATCH --job-name=sopt-e7-smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -euo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/vla_train.py --arm expert_droid --k 5 --run e7_smoke --steps 200 --bs 32 --workers 8
python scripts/vla_eval.py --ckpt outputs/e7_smoke/final.pt --out outputs/e7_smoke/rollouts.json --n-init 1 --max-steps 60
echo "end=$(date -Is)"
