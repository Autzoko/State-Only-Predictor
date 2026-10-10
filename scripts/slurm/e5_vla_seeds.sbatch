#!/bin/bash
# E5 extra training seeds (k=5): SmolVLA none / scratch / droid x seeds 1, 2 (one GPU per arm, two waves), then eval.
#   sbatch scripts/slurm/e5_vla_seeds.sbatch
#SBATCH --job-name=sopt-e5-seeds
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
for seed in 1 2; do
  gpu=0
  for arm in none scratch droid; do
    name=e5_vla_${arm}_k5_s$seed
    [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_train.py --arm $arm --k 5 --seed $seed \
      --run $name --steps 20000 --bs 32 --workers 8 > logs/$name.log 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  gpu=0
  for arm in none scratch droid; do
    name=e5_vla_${arm}_k5_s$seed
    [ -f outputs/$name/rollouts.json ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_eval.py \
      --ckpt outputs/$name/final.pt --out outputs/$name/rollouts.json > logs/$name.eval.log 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  echo "seed $seed done $(date -Is)"
done
echo "end=$(date -Is)"
