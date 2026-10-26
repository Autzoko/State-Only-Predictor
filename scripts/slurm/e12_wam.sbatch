#!/bin/bash
# E12: FastWAM (world action model) + SOPT, k5. One seed per job (SEED env), three arms in parallel on 3 GPUs:
# none / expert_scratch / expert_droid. 10k steps (bs 16, ~2.3 h), then 50-init closed-loop eval (~3 h).
#   SEED=0 sbatch scripts/slurm/e12_wam.sbatch
# E38 prior instances (2 GPUs): SBATCH_GRES=gpu:a100:2 SUITE=libero90_17 SEED=0 PSET=_pB1=outputs/e25j_base/best.pt,_pB2=outputs/e25j_base2/best.pt sbatch scripts/slurm/e12_wam.sbatch
#SBATCH --job-name=sopt-e12
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
S=${SEED:-0}; SUITE=${SUITE:-libero90}; ST=$([ "$SUITE" = libero90_17 ] && echo _t17)
PRIOR=${PRIOR:-outputs/e2_ar_flow_f100_s/best.pt}; PT=${PTAG:-}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) seed=$S pset=${PSET:-} start=$(date -Is)"
run() { local name=e12_wam_$1_k5${ST}${PT}_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/wam_train.py --arm $1 --k 5 --seed $S --suite $SUITE \
    --run $name --steps 10000 --bs 16 --workers 8 --prior $PRIOR > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/wam_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
if [ -n "${PSET:-}" ]; then  # E38: expert_droid per prior instance, one GPU each (PSET=_pB1=<ckpt>,_pB2=<ckpt>)
  g=0; for x in ${PSET//,/ }; do (PT=${x%%=*}; PRIOR=${x#*=}; [ -f "$PRIOR" ] || { echo "missing prior $PRIOR"; exit 1; }
    run expert_droid $g) & g=$((g+1)); done; wait
else run none 0 & run expert_scratch 1 & run expert_droid 2 & wait; fi
echo "end=$(date -Is)"
