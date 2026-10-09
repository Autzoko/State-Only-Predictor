#!/usr/bin/env bash
# E4b multi-seed: training seeds 1 and 2 (seed 0 = the v1/v2 runs) for BC / trunk-BC scratch / trunk-BC DROID,
# k=5 and all demos. Training 4 concurrent; closed-loop evals one at a time (GPU memory is shared).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DROID=outputs/e2_ar_flow_f100_s/best.pt
runs=()
for seed in 1 2; do for k in 5 0; do
  tag=$([ $k = 0 ] && echo all || echo k$k)
  runs+=("e4b_bc_${tag}_s$seed|--kind bc --k $k --seed $seed")
  runs+=("e4b_trunkbc_scratch_${tag}_s$seed|--kind trunkbc --k $k --init scratch --seed $seed")
  runs+=("e4b_trunkbc_droid_${tag}_s$seed|--kind trunkbc --k $k --init $DROID --seed $seed")
done; done
train() { local name=${1%%|*} args=${1#*|}; [ -f outputs/$name/final.pt ] && return
  python scripts/e4b_train.py --run $name $args --steps 15000 --workers 4 > outputs/$name.log 2>&1; }
for i in $(seq 0 4 $((${#runs[@]} - 1))); do
  for j in 0 1 2 3; do [ $((i+j)) -lt ${#runs[@]} ] && train "${runs[$((i+j))]}" & done; wait
done
echo "TRAIN DONE $(date -Is)"
specs=(); for r in "${runs[@]}"; do specs+=("${r%%|*}:1"); done
bash scripts/remote/e4b_eval_queue.sh "${specs[@]}"
