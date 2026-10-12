#!/bin/bash
# E7b: (1) traced SmolVLA rollouts (none vs expert_droid, seed 0) for the failure taxonomy;
# (2) the E7 comparison with all demos (none / expert_scratch / expert_droid, seeds 0-2), 50 init states.
#   sbatch scripts/slurm/e7b_vla.sbatch
#SBATCH --job-name=sopt-e7b
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
train() { [ -f outputs/$1/final.pt ] || CUDA_VISIBLE_DEVICES=$3 python scripts/vla_train.py --arm $2 --k 0 --seed $4 \
  --run $1 --steps 20000 --bs 32 --workers 8 > logs/$1.log 2>&1; }
evaluate() { [ -f outputs/$1/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
  --ckpt outputs/$1/final.pt --out outputs/$1/rollouts_n50.json --n-init 50 > logs/$1.eval_n50.log 2>&1; }
trace() { [ -f outputs/$1/trace_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
  --ckpt outputs/$1/final.pt --out outputs/$1/trace_n50.json --n-init 50 --trace 4 > logs/$1.trace_n50.log 2>&1; }
trace e5_vla_none_k5 0 & trace e7_vla_expert_droid_k5_s0 1 &
train e7b_vla_none_all_s0 none 2 0 & wait
echo "traces $(date -Is)"
train e7b_vla_expert_droid_all_s0 expert_droid 0 0 & train e7b_vla_expert_scratch_all_s0 expert_scratch 1 0 &
train e7b_vla_none_all_s1 none 2 1 & wait
echo "wave 2 $(date -Is)"
train e7b_vla_expert_droid_all_s1 expert_droid 0 1 & train e7b_vla_expert_scratch_all_s1 expert_scratch 1 1 &
train e7b_vla_none_all_s2 none 2 2 & wait
echo "wave 3 $(date -Is)"
train e7b_vla_expert_droid_all_s2 expert_droid 0 2 & train e7b_vla_expert_scratch_all_s2 expert_scratch 1 2 & wait
echo "wave 4 $(date -Is)"
runs=(e7b_vla_none_all_s0 e7b_vla_expert_droid_all_s0 e7b_vla_expert_scratch_all_s0
      e7b_vla_none_all_s1 e7b_vla_expert_droid_all_s1 e7b_vla_expert_scratch_all_s1
      e7b_vla_none_all_s2 e7b_vla_expert_droid_all_s2 e7b_vla_expert_scratch_all_s2)
for i in 0 3 6; do
  evaluate ${runs[$i]} 0 & evaluate ${runs[$((i+1))]} 1 & evaluate ${runs[$((i+2))]} 2 & wait
  echo "eval round $i $(date -Is)"
done
echo "end=$(date -Is)"
