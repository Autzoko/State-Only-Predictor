#!/usr/bin/env bash
# E4b v2: prior Transformer as BC trunk (scratch / DROID / DROID->LIBERO), plus prior+IDM with K=16 mean plans.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DROID=outputs/e2_ar_flow_f100_s/best.pt
DROID_LIBERO=outputs/e4a_ar_flow_ft_kall/best.pt
runs=()
for k in 5 0; do
  tag=$([ $k = 0 ] && echo all || echo k$k)
  runs+=("e4b_trunkbc_scratch_$tag|--kind trunkbc --k $k --init scratch")
  runs+=("e4b_trunkbc_droid_$tag|--kind trunkbc --k $k --init $DROID")
  runs+=("e4b_trunkbc_droidlibero_$tag|--kind trunkbc --k $k --init $DROID_LIBERO")
done
train() { local name=${1%%|*} args=${1#*|}; [ -f outputs/$name/final.pt ] && return
  python scripts/e4b_train.py --run $name $args --steps 15000 --workers 5 > outputs/$name.log 2>&1; }
evaluate() { local name=$1 extra=${2:-} out=${3:-rollouts.json}; [ -f outputs/$name/$out ] && return
  python scripts/e4b_eval.py --policy outputs/$name/final.pt --idm outputs/e4b_idm/final.pt --n-init 20 $extra \
    --out outputs/$name/$out > outputs/$name.eval_${out%.json}.log 2>&1; }
for i in 0 3; do for j in 0 1 2; do train "${runs[$((i+j))]}" & done; wait; done
echo "TRAIN DONE $(date -Is)"
names=(); for r in "${runs[@]}"; do names+=("${r%%|*}"); done
for i in 0 2 4; do evaluate "${names[$i]}" & evaluate "${names[$((i+1))]}" & wait; done
evaluate e4b_prior_droid_k5 "--plan-samples 16" rollouts_k16.json & evaluate e4b_prior_droid_all "--plan-samples 16" rollouts_k16.json & wait
echo "ALL DONE $(date -Is)"
