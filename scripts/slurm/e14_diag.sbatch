#!/bin/bash
# E14 diagnosis: open-loop action error (train demos vs held-out demos) of pi0.5 and SmolVLA k5 policies.
#SBATCH --job-name=sopt-e14d
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
for c in e5_vla_none_k5 e14_pi05_none_k5_s0 e14_pi05_expert_droid_k5_s0; do
  python scripts/vla_openloop.py --ckpt outputs/$c/final.pt 2>&1 | grep -E "^(train|heldout|\{)|Error|error" | tail -4
done
echo "end=$(date -Is)"
