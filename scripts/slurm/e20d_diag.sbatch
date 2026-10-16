#!/bin/bash
# E20d: why SOPT hurts X-VLA (scripts/vla_diag.py): implementation checks + counterfactual open-loop probes of the
# 17-task X-VLA checkpoints (s0, s1), with DP (SOPT helps) and GR00T (no gain) s0 as contrasts. One GPU, ~1 h.
#   sbatch scripts/slurm/e20d_diag.sbatch
#SBATCH --job-name=sopt-e20d
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
C=()
for h in e20_xvla_%s_k5_t17_s0 e20_xvla_%s_k5_t17_s1 e21_dp_%s_k5_t17_s0 e22_groot_%s_k5_t17_s0; do
  for a in expert_droid none expert_scratch; do C+=(outputs/$(printf $h $a)/final.pt); done
done
python scripts/vla_diag.py --out outputs/e20d_diag.json --init-check --ckpts "${C[@]}" > logs/e20d_diag.log 2>&1
echo "diag exit $? end=$(date -Is)"; tail -3 logs/e20d_diag.log
