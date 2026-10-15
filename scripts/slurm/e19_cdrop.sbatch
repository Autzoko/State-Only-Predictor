#!/bin/bash
# E19: SOPT dropout ("do no harm"): SmolVLA + DROID SOPT (expert_droid), the SOPT feature zeroed per training sample
# with probability CDROP (default 0.5); evaluation always uses it. Seeds 0-2 in parallel, k5. Everything else equals
# the paired runs it is compared with:
#   SUITE=libero90 (default): E7 (e5_vla_none_k5*, e7_vla_expert_droid_k5_s*), EE+joint prior e2 (as E7)
#   SUITE=libero_spatial / ...: E13 k5 (e13_<suite>_{none,expert_droid}_k5_s*), EE-only prior e13 (as E13)
#   SUITE=libero_spatial sbatch scripts/slurm/e19_cdrop.sbatch
#SBATCH --job-name=sopt-e19
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
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
SUITE=${SUITE:-libero90}; CD=${CDROP:-0.5}; CDT=cd${CD/./}
if [ "$SUITE" = libero90 ]; then PRIOR=outputs/e2_ar_flow_f100_s/best.pt; PFX=e19_vla; else
  PRIOR=outputs/e13_ar_flow_f100_s_ee/best.pt; PFX=e19_$SUITE; fi
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) suite=$SUITE cdrop=$CD start=$(date -Is)"
run() { local name=${PFX}_expert_droid_${CDT}_k5_s$1
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$1 python scripts/vla_train.py --suite $SUITE --prior $PRIOR \
    --arm expert_droid --k 5 --seed $1 --run $name --steps 20000 --bs 32 --workers 8 --cond-drop $CD > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$1 python scripts/vla_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
run 0 & run 1 & run 2 & wait
echo "end=$(date -Is)"
