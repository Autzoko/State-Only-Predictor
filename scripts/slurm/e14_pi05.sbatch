#!/bin/bash
# E14: pi0.5-DROID (VLA already trained on DROID actions) + SOPT, LIBERO-90 E4b tasks, k5 (same protocol as E7 / E12).
# One seed per job (SEED env); arms none / expert_scratch / expert_droid in parallel on 3 GPUs. A 30-step smoke run
# (train + 1-init eval) of expert_droid goes first and aborts the job if it fails.
#   SEED=0 sbatch scripts/slurm/e14_pi05.sbatch
#SBATCH --job-name=sopt-e14
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
S=${SEED:-0}; STEPS=${STEPS:-20000}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) seed=$S steps=$STEPS start=$(date -Is)"
if [ "$S" = 0 ] && [ ! -f outputs/e14_smoke/rollouts.json ]; then
  python scripts/vla_train.py --vla pi05 --arm expert_droid --k 5 --seed 0 --run e14_smoke --steps 30 --bs 32 \
    --workers 8 > logs/e14_smoke.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e14_smoke/final.pt --out outputs/e14_smoke/rollouts.json --n-init 1 \
    --max-steps 40 > logs/e14_smoke.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv
  [ $rc = 0 ] || exit 1
fi
run() { local name=e14_pi05_$1_k5_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_train.py --vla pi05 --arm $1 --k 5 \
    --seed $S --run $name --steps $STEPS --bs 32 --workers 8 > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
run none 0 & run expert_scratch 1 & run expert_droid 2 & wait
echo "end=$(date -Is)"
