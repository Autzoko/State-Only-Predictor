#!/bin/bash
# E13: standard LIBERO benchmark suite (SUITE env), SmolVLA + EE-only SOPT, all demos, seed SEED (default 0).
# Arms in parallel: none / expert_scratch / expert_droid (EE-only SOPT pretrained on DROID only, e13_ar_flow_f100_s_ee).
# 20k steps (bs 32), then 50 init states x 10 tasks = 500 episodes per arm (OpenVLA step limits).
#   SUITE=libero_spatial sbatch scripts/slurm/e13_libero_std.sbatch
#SBATCH --job-name=sopt-e13
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
S=${SEED:-0}; PRIOR=outputs/e13_ar_flow_f100_s_ee/best.pt
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) suite=$SUITE seed=$S start=$(date -Is)"
python scripts/prepare_libero_std.py --suite $SUITE --sopt-data $SOPT_DATA --workers 24 > logs/prep_$SUITE.log 2>&1
echo "prep exit $? $(date -Is)"; tail -n 1 logs/prep_$SUITE.log
export HF_HUB_OFFLINE=1
run() { local name=e13_${SUITE}_$1_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_train.py --suite $SUITE --prior $PRIOR \
    --arm $1 --k 0 --seed $S --run $name --steps 20000 --bs 32 --workers 8 > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
run none 0 & run expert_scratch 1 & run expert_droid 2 & wait
echo "end=$(date -Is)"
