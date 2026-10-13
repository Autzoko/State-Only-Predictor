#!/bin/bash
# E15 diagnostic: why does SOPT not help on the standard LIBERO suites (E13) while it helps on LIBERO-90 (E7 / E7b)?
# Rerun E7 exactly (SmolVLA, LIBERO-90 E4b tasks, k5, seeds 0-2, 50 inits) with the EE-only DROID prior used in
# E13 (e13_ar_flow_f100_s_ee: joints hidden inside SOPT; same normalizer as e2_ar_flow_f100_s, checked). Everything
# else, including SmolVLA's current-state token, is identical to E7, so E7 none / expert_droid are the paired controls.
#   sbatch scripts/slurm/e15_ee_prior.sbatch
#SBATCH --job-name=sopt-e15
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
run() { local name=e15_vla_expert_droid_ee_k5_s$1
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$1 python scripts/vla_train.py --arm expert_droid --k 5 \
    --seed $1 --prior outputs/e13_ar_flow_f100_s_ee/best.pt --run $name --steps 20000 --bs 32 --workers 8 > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$1 python scripts/vla_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
run 0 & run 1 & run 2 & wait
echo "end=$(date -Is)"
