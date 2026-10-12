#!/usr/bin/env bash
# E10: keyframe ("where") + pretrained goal-conditioned SOPT ("how"), k5, 3 seeds, 50 init states.
# Arms: gc-pretrained prior (DROID + DROID hindsight keyframes, E3a), DROID prior with a fresh goal pathway,
# scratch prior. Baselines (BC, TrunkBC scratch / DROID; existing E4b checkpoints) re-evaluated at 50 inits.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
evaluate() { [ -f outputs/$1/rollouts_n50.json ] && return
  python scripts/e4b_eval.py --policy outputs/$1/final.pt --idm outputs/e4b_idm/final.pt --n-init 50 \
    --out outputs/$1/rollouts_n50.json > outputs/$1.eval_n50.log 2>&1; echo "$1 eval exit $? $(date -Is)"; }
train() { [ -f outputs/$1/final.pt ] && return
  python scripts/e4b_train.py --run $1 --kind kfmotor --k 5 --steps 15000 --workers 4 "${@:2}" > outputs/$1.log 2>&1
  echo "$1 train exit $? $(date -Is)"; }
for seed in 0 1 2; do
  train e10_kf_gcdroid_k5_s$seed --init outputs/gc_ar_flow_ft/best.pt --seed $seed
  evaluate e10_kf_gcdroid_k5_s$seed
  train e10_kf_scratch_k5_s$seed --init scratch --seed $seed
  evaluate e10_kf_scratch_k5_s$seed
done
for s in "" _s1 _s2; do
  for b in e4b_bc_k5 e4b_trunkbc_droid_k5 e4b_trunkbc_scratch_k5; do evaluate $b$s; done
done
for seed in 0 1 2; do
  train e10_kf_droid_k5_s$seed --init outputs/e2_ar_flow_f100_s/best.pt --seed $seed
  evaluate e10_kf_droid_k5_s$seed
done
echo "E10 DONE $(date -Is)"
