#!/bin/bash
# E21: Diffusion Policy (ResNet-18 ImageNet + 1-D U-Net; no robot pretraining) + SOPT, LIBERO-90 E4b tasks, k5 (protocol
# of E7 / E12 / E14 / E20). One seed per job (SEED env); arms none / expert_scratch / expert_droid in parallel on 3 GPUs;
# 30k steps at batch 64 (Diffusion Policy needs more updates than the VLAs). Smoke run first for seed 0.
#   SEED=0 sbatch scripts/slurm/e21_dp.sbatch      (STEPS, LR, TAG, NEXECS, EVAL_ONLY as in e14_pi05.sbatch)
# E38 prior instances: one arm on one GPU with another SOPT checkpoint (run names get PTAG before _s<seed>; paired
# with the existing none arm of the same seed / demos / inits):
#   SBATCH_GRES=gpu:a100:1 SBATCH_MEM_PER_NODE=120G SUITE=libero90_17 SEED=0 ARM=expert_droid \
#     PRIOR=outputs/e25j_base/best.pt PTAG=_pB1 sbatch scripts/slurm/e21_dp.sbatch
#SBATCH --job-name=sopt-e21
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=/scratch/ll5582/SOPT_DATA/torch_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
SUITE=${SUITE:-libero90}; ST=$([ "$SUITE" = libero90_17 ] && echo _t17); S=${SEED:-0}; STEPS=${STEPS:-30000}; TAG=${TAG:-}; NEXECS=${NEXECS:-10}; LRARG=${LR:+--lr $LR}
PRIOR=${PRIOR:-outputs/e2_ar_flow_f100_s/best.pt}; PT=${PTAG:-}
[ -f "$PRIOR" ] || { echo "missing prior $PRIOR"; exit 1; }
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) suite=$SUITE seed=$S steps=$STEPS prior=$PRIOR start=$(date -Is)"
if [ "$S" = 0 ] && [ ! -f outputs/e21_smoke/rollouts.json ] && [ -z "${EVAL_ONLY:-}" ]; then
  python scripts/vla_train.py --vla diffusion --arm expert_droid --k 5 --seed 0 --run e21_smoke --steps 30 --bs 64 \
    --workers 8 > logs/e21_smoke.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e21_smoke/final.pt --out outputs/e21_smoke/rollouts.json --n-init 1 \
    --max-steps 40 > logs/e21_smoke.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; nvidia-smi --query-gpu=memory.used --format=csv
  [ $rc = 0 ] || exit 1
fi
run() { local name=e21_dp_$1_k5${TAG}${ST}${PT}_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_train.py --vla diffusion --arm $1 --k 5 --suite $SUITE \
    --seed $S --run $name --steps $STEPS --bs 64 --workers 8 --prior $PRIOR $LRARG > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  for n in $NEXECS; do local x=$([ "$n" = 10 ] || echo _x$n)
    [ -f outputs/$name/rollouts_n50$x.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py --n-exec $n \
      --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50$x.json --n-init 50 > logs/$name.eval_n50$x.log 2>&1
    echo "$name eval n_exec $n exit $? $(date -Is)"
  done; }
if [ -n "${ARM:-}" ]; then run $ARM 0
else run none 0 & run expert_scratch 1 & run expert_droid 2 & wait; fi
echo "end=$(date -Is)"
