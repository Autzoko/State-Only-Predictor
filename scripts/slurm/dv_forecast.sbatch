#!/bin/bash
# Data validation DV1 (history length) + DV3 (goal decomposition) on DROID held-out labs. Pre-registered in
# docs/EXPERIMENTS.md ("数据验证 DV1–DV3"). DROID states only; checkpoints selected on val (UW), numbers on test.
#   DV1  train ctx4 / ctx16 priors (e2 recipe, seed 0, = e25j_ctx48 with model.ctx_len changed), then score
#        ctx 4 / 16 / 48 / 96 (+ the two other ctx96 instances) on one fixed anchor set (96 real history frames).
#        Secondary: anchor set with 384 history frames adds ctx192 / ctx384; val split replication.
#   DV3  E3a goal-conditioned priors (gc_ar_flow_ft, gc_ar_flow_scratch; copied from 127, md5 checked):
#        per-step energy score with no goal / keyframe goal / endpoint goal on the same anchors.
#   sbatch scripts/slurm/dv_forecast.sbatch
#SBATCH --job-name=sopt-dv
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4
D=$SOPT_DATA/processed/droid
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
for c in 4 16; do
  P=outputs/dv1_ctx$c
  [ -f $P/best.pt ] || { python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e2.yaml \
    model.objective=ar_flow data.root=$D run_name=dv1_ctx$c train.num_workers=8 model.ctx_len=$c \
    > logs/dv1_ctx$c.log 2>&1; echo "dv1_ctx$c pretrain exit $? $(date -Is)"; tail -n 2 logs/dv1_ctx$c.log; }
  [ -f $P/best.pt ] || exit 1
done
for f in outputs/e2_ar_flow_f100_s outputs/e25j_ctx48 outputs/e25j_base outputs/e25j_base2 outputs/e25j_ctx192 \
         outputs/e25j_ctx384 outputs/gc_ar_flow_ft outputs/gc_ar_flow_scratch; do
  [ -f $f/best.pt ] || { echo "missing $f/best.pt"; exit 1; }
done
M96="ctx4=outputs/dv1_ctx4/best.pt ctx16=outputs/dv1_ctx16/best.pt ctx48=outputs/e25j_ctx48/best.pt \
ctx96=outputs/e2_ar_flow_f100_s/best.pt ctx96_b1=outputs/e25j_base/best.pt ctx96_b2=outputs/e25j_base2/best.pt"
O=outputs/dv1; mkdir -p $O
for sp in test val; do
  [ -f $O/a96_$sp.npz ] || { python scripts/dv_anchor_eval.py --split $sp --anchor-ctx 96 --data-root $D \
    --ckpt $M96 --out $O/a96_$sp.npz > $O/a96_$sp.log 2>&1; echo "dv1 a96 $sp exit $? $(date -Is)"; }
done
[ -f $O/a384_test.npz ] || { python scripts/dv_anchor_eval.py --split test --anchor-ctx 384 --data-root $D \
  --ckpt $M96 ctx192=outputs/e25j_ctx192/best.pt ctx384=outputs/e25j_ctx384/best.pt --out $O/a384_test.npz \
  > $O/a384_test.log 2>&1; echo "dv1 a384 test exit $? $(date -Is)"; }
O=outputs/dv3; mkdir -p $O
for r in gc_ar_flow_ft gc_ar_flow_scratch; do
  [ -f $O/${r}_test.npz ] || { python scripts/dv_anchor_eval.py --split test --anchor-ctx 96 --data-root $D \
    --goal-modes none endpoint keyframe --ckpt $r=outputs/$r/best.pt --out $O/${r}_test.npz \
    > $O/${r}_test.log 2>&1; echo "dv3 $r exit $? $(date -Is)"; }
done
for f in outputs/dv1/*.json outputs/dv3/*.json; do echo "== $f"; python -c "import json,sys; d=json.load(open('$f')); \
print(d['windows'], d['episodes'], {k: round(v, 3) for k, v in d['mean_es'].items()})"; done
echo "end=$(date -Is)"
