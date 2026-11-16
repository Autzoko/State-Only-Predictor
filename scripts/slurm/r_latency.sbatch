#!/bin/bash
# Inference cost of SOPT (reviewer request, 2026-11-16; scripts/latency.py): batch-1 action-chunk latency with / without
# SOPT on DP, SmolVLA and GR00T (seed-0 main-table checkpoints) and the SOPT encoder alone, one A100 80GB.
#   sbatch scripts/slurm/r_latency.sbatch
#SBATCH --job-name=sopt-rlat
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=/scratch/ll5582/SOPT_DATA/torch_cache
export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"; nvidia-smi --query-gpu=name --format=csv,noheader
C=""; for r in e21_dp_none_k5_t17_s0 e21_dp_expert_droid_k5_t17_s0 e23_smolvla_none_k5_jr_t17_s0 \
  e23_smolvla_expert_droid_k5_jr_t17_s0 e22_groot_none_k5_t17_s0 e31_groot_xattn_droid_k5_t17_s0; do C="$C outputs/$r/final.pt"; done
python scripts/latency.py --ckpts $C --out outputs/r_latency.json > logs/r_latency.log 2>&1
echo "latency exit $? $(date -Is)"; tail -8 logs/r_latency.log
