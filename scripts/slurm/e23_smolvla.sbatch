#!/bin/bash
# E23: SmolVLA + SOPT on the 17-task LIBERO-90 protocol (7 E4b test tasks + 10 val-split tasks; configs/libero90_17_tasks.json),
# E7 recipe. One seed per job; arms none / expert_scratch / expert_droid in parallel on 3 GPUs.
#   SUITE=libero90_17 SEED=0 sbatch scripts/slurm/e23_smolvla.sbatch
# One arm on one GPU (1-GPU jobs start much sooner; idle GPUs get jobs cancelled):
#   SBATCH_GRES=gpu:a100:1 SBATCH_MEM_PER_NODE=120G SUITE=libero90_17 SEED=2 ARM=none sbatch scripts/slurm/e23_smolvla.sbatch
#SBATCH --job-name=sopt-e23
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
SUITE=${SUITE:-libero90}; ST=$([ "$SUITE" = libero90_17 ] && echo _t17); S=${SEED:-0}; STEPS=${STEPS:-20000}; TAG=${TAG:-}; NEXECS=${NEXECS:-10}; LRARG=${LR:+--lr $LR}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) suite=$SUITE seed=$S steps=$STEPS start=$(date -Is)"
if [ "$S" = 0 ] && [ ! -f outputs/e23_smoke/rollouts.json ] && [ -z "${EVAL_ONLY:-}" ]; then
  python scripts/vla_train.py --vla smolvla --arm expert_droid --k 5 --seed 0 --run e23_smoke --steps 30 --bs 32 \
    --workers 8 > logs/e23_smoke.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e23_smoke/final.pt --out outputs/e23_smoke/rollouts.json --n-init 1 \
    --max-steps 40 > logs/e23_smoke.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv
  [ $rc = 0 ] || exit 1
fi
run() { local name=e23_smolvla_$1_k5${TAG}${ST}_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_train.py --vla smolvla --arm $1 --k 5 --suite $SUITE \
    --seed $S --run $name --steps $STEPS --bs 32 --workers 8 $LRARG > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  for n in $NEXECS; do local x=$([ "$n" = 10 ] || echo _x$n)
    [ -f outputs/$name/rollouts_n50$x.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py --n-exec $n \
      --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50$x.json --n-init 50 > logs/$name.eval_n50$x.log 2>&1
    echo "$name eval n_exec $n exit $? $(date -Is)"
  done; }
if [ -n "${ARM:-}" ]; then run $ARM 0
else run none 0 & run expert_scratch 1 & run expert_droid 2 & wait; fi
echo "end=$(date -Is)"
