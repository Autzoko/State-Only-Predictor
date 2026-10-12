#!/bin/bash
# E7: SOPT feature conditions SmolVLA's action expert (arms expert_droid / expert_scratch), k=5, seeds 0-2.
# Evaluation on 50 init states per task (350 paired episodes per seed) for every arm, including the E5 `none`
# checkpoints (same training recipe; E7 code leaves the none path unchanged).
#   sbatch scripts/slurm/e7_vla_expert.sbatch
#SBATCH --job-name=sopt-e7
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
train() { [ -f outputs/$1/final.pt ] || CUDA_VISIBLE_DEVICES=$3 python scripts/vla_train.py --arm $2 --k 5 --seed $4 \
  --run $1 --steps 20000 --bs 32 --workers 8 > logs/$1.log 2>&1; }
evaluate() { [ -f outputs/$1/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
  --ckpt outputs/$1/final.pt --out outputs/$1/rollouts_n50.json --n-init 50 > logs/$1.eval_n50.log 2>&1; }
train e7_vla_expert_droid_k5_s0 expert_droid 0 0 & train e7_vla_expert_scratch_k5_s0 expert_scratch 1 0 &
train e7_vla_expert_droid_k5_s1 expert_droid 2 1 & wait
echo "wave 1 $(date -Is)"
train e7_vla_expert_scratch_k5_s1 expert_scratch 0 1 & train e7_vla_expert_droid_k5_s2 expert_droid 1 2 &
train e7_vla_expert_scratch_k5_s2 expert_scratch 2 2 & wait
echo "wave 2 $(date -Is)"
runs=(e5_vla_none_k5 e7_vla_expert_droid_k5_s0 e7_vla_expert_scratch_k5_s0
      e5_vla_none_k5_s1 e7_vla_expert_droid_k5_s1 e7_vla_expert_scratch_k5_s1
      e5_vla_none_k5_s2 e7_vla_expert_droid_k5_s2 e7_vla_expert_scratch_k5_s2)
for i in 0 3 6; do
  evaluate ${runs[$i]} 0 & evaluate ${runs[$((i+1))]} 1 & evaluate ${runs[$((i+2))]} 2 & wait
  echo "eval round $i $(date -Is)"
done
echo "end=$(date -Is)"
