#!/bin/bash
# E33 on Jubail (DGX unreachable since 2026-10-17): EE-only DROID priors with other history lengths, for the
# long-horizon host runs (E35, libero_10). Recipe = e13_ar_flow_f100_s_ee (S, ar_flow, 100% DROID, ee_only, 40k
# steps) with only model.ctx_len changed: ctx48 (3.2 s) | ctx192 (12.8 s) | ctx384 (25.6 s), one per GPU.
# Longer contexts use data.left_pad = ctx - 96, so every window keeps >= 96 real frames: the same episodes / anchors
# as ctx96, only older history added (first-frame padding, as LIBERO rollouts pad early history). DROID only.
#   sbatch scripts/slurm/e33_prior.sbatch
# One prior on one GPU (e.g. a ctx96 replica of e13 trained with this script, to separate history length from
# prior-run variance): SBATCH_GRES=gpu:a100:1 SBATCH_MEM_PER_NODE=80G ONE=96 sbatch scripts/slurm/e33_prior.sbatch
#SBATCH --job-name=sopt-e33
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA OPENBLAS_NUM_THREADS=4
D=$SOPT_DATA/processed/droid
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
[ -f $D/episodes.parquet ] || { echo "no DROID states in $D"; exit 1; }
run() { local n=$1 gpu=$2 pad=$3 name=e33_ar_flow_f100_s_ee_ctx$1
  [ -f outputs/$name/best.pt ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py --config configs/model/s.yaml \
    --config configs/experiment/e2.yaml model.objective=ar_flow model.ee_only=true data.root=$D run_name=$name \
    model.ctx_len=$n data.left_pad=$pad train.num_workers=8 > logs/$name.log 2>&1
  echo "$name exit $? $(date -Is)"; tail -2 logs/$name.log; }
if [ -n "${ONE:-}" ]; then run $ONE 0 $(( ONE > 96 ? ONE - 96 : 0 ))
else run 48 0 0 & run 192 1 96 & run 384 2 288 & wait; fi
echo "end=$(date -Is)"
