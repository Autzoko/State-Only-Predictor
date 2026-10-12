#!/usr/bin/env bash
# E11: does more / bigger state-only pretraining give a better downstream policy? (TrunkBC, k5, 3 seeds, 50 inits)
#   pretraining data: DROID 1% / 10% / 100% (existing S checkpoints e2_ar_flow_f{001,010,100}_s)
#   model size:       S (5M) vs B (d512 x 12 layers, pretrained here on 100% DROID, same recipe, lr 2e-4)
# Starts after E8 finishes (one GPU on 127).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
until grep -q "E8 DONE" outputs/e8_queue.log 2>/dev/null; do sleep 300; done
echo "start $(date -Is)"
evaluate() { [ -f outputs/$1/rollouts_n50.json ] && return
  python scripts/e4b_eval.py --policy outputs/$1/final.pt --idm outputs/e4b_idm/final.pt --n-init 50 \
    --out outputs/$1/rollouts_n50.json > outputs/$1.eval_n50.log 2>&1; echo "$1 eval exit $? $(date -Is)"; }
train() { [ -f outputs/$1/final.pt ] && return
  python scripts/e4b_train.py --run $1 --kind trunkbc --k 5 --steps 15000 --workers 4 --init $2 --seed $3 \
    > outputs/$1.log 2>&1; echo "$1 train exit $? $(date -Is)"; }
for seed in 0 1 2; do
  for f in 001 010; do
    train e11_trunk_f${f}_k5_s$seed outputs/e2_ar_flow_f${f}_s/best.pt $seed; evaluate e11_trunk_f${f}_k5_s$seed
  done
done
if [ ! -f outputs/e11_ar_flow_f100_b/best.pt ]; then
  python scripts/train.py --config configs/model/b.yaml --config configs/experiment/e2.yaml model.objective=ar_flow \
    train.lr=2e-4 run_name=e11_ar_flow_f100_b > outputs/e11_ar_flow_f100_b.log 2>&1
  echo "pretrain B exit $? $(date -Is)"
fi
for seed in 0 1 2; do
  train e11_trunk_b_k5_s$seed outputs/e11_ar_flow_f100_b/best.pt $seed; evaluate e11_trunk_b_k5_s$seed
done
echo "E11 DONE $(date -Is)"
