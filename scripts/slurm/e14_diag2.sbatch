#!/bin/bash
# E14 diagnosis 2: closed-loop sensitivity of pi0.5 (none, s0) to actions executed per chunk + traced failures.
#SBATCH --job-name=sopt-e14d
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
C=outputs/e14_pi05_none_k5_s0
python scripts/vla_eval.py --ckpt $C/final.pt --out $C/diag_trace10.json --n-init 10 --trace 4 > logs/e14d_trace.log 2>&1
echo "trace exit $?"; python scripts/failure_taxonomy.py $C/diag_trace10.json
for n in 5 25 50; do
  python scripts/vla_eval.py --ckpt $C/final.pt --out $C/diag_exec$n.json --n-init 10 --n-exec $n > logs/e14d_exec$n.log 2>&1
  echo "n_exec $n exit $? $(grep -m1 success_rate $C/diag_exec$n.json)"
done
echo "end=$(date -Is)"
