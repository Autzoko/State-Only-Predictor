#!/bin/bash
# E34: is the SOPT history what makes the strong hosts carry on after a missed grasp ("phantom transport", E26)?
# Evaluation only, no training: switch the SOPT input off for 2 s after every gripper-close command and compare with the
# same checkpoint unmasked (paired by task + init; traces + action log -> e26_taxonomy.py / analysis/e26_followup.py).
#   gpu0  e22_groot_expert_droid s0           masked 2 s       (vs E26 trace; trained without dropout -> masking is OOD)
#   gpu1  e31_groot_xattn_droid_cd03_plr01 s0 masked 2 s       (trained with branch dropout 0.3 -> masking in-distribution)
#   gpu2  e31_groot_xattn_droid_cd03_plr01 s0 unmasked trace, then SOPT always off (does the branch help at all?)
#   sbatch scripts/slurm/e34_mask.sbatch            (after E31 SET=screen)
#SBATCH --job-name=sopt-e34
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
D=outputs/e22_groot_expert_droid_k5_t17_s0; X=outputs/e31_groot_xattn_droid_k5_cd03_plr01_t17_s0
[ -f $X/final.pt ] || { echo "missing $X/final.pt (run E31 SET=screen first)"; exit 1; }
# smoke: masking path of both interfaces (expert: feature zeroed; xattn: branch keep mask)
for c in $D $X; do
  python scripts/vla_eval.py --ckpt $c/final.pt --out /tmp/e34_smoke_$$.json --n-init 1 --max-steps 40 --sopt-mask-grasp 2 \
    > logs/e34_smoke.log 2>&1 || { echo "smoke $c failed"; tail -5 logs/e34_smoke.log; exit 1; }
done
echo "smoke ok $(date -Is)"
ev() { local gpu=$1 ck=$2 tag=$3 mask=$4
  [ -f $ck/trace${tag}_n50.json ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_eval.py --ckpt $ck/final.pt \
    --out $ck/trace${tag}_n50.json --n-init 50 --trace 4 --act-log --sopt-mask-grasp $mask > logs/e34_$(basename $ck)$tag.log 2>&1
  echo "$(basename $ck)$tag exit $? $(date -Is)"; }
ev 0 $D _mask2 2 &
ev 1 $X _mask2 2 &
{ ev 2 $X "" 0; ev 2 $X _maskall -1; } &
wait
echo "end=$(date -Is)"
