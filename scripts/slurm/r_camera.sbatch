#!/bin/bash
# R1 / R2 (reviewer controls, preregistered 2026-11-16): camera-shift evaluations of existing checkpoints (no training),
# clean results = the runs' rollouts_n50.json. Used for the scratch arms of DP / SmolVLA-goal at 10 and 30 deg (their
# 20 deg evaluations exist from E37s), so every SOPT-structure arm has the full camera dose.
#   RUNS="e21_dp_expert_scratch_k5_t17_s0 e21_dp_expert_scratch_k5_t17_s1" CAM_LEVELS="1 3" sbatch scripts/slurm/r_camera.sbatch
#SBATCH --job-name=sopt-rcam
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/SOPT/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) runs=${RUNS:?} levels=${CAM_LEVELS:-1 2 3} start=$(date -Is)"
source scripts/slurm/cam_evals.sh
for r in $RUNS; do [ -f outputs/$r/rollouts_n50.json ] || { echo "missing clean rollouts of $r"; exit 1; }; done
for r in $RUNS; do cam_evals $r 0; done
echo "end=$(date -Is)"
