#!/usr/bin/env bash
# E6: pretrained prior as the action decoder (MotionDecoder), 3 seeds. k5 first (train -> eval), then all demos.
# Arms: DROID init vs scratch init (both with the foresight loss); k5 also DROID without the foresight loss.
# Training 3 concurrent (GPU is shared, ~17 GB free); closed-loop evals one at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DROID=outputs/e2_ar_flow_f100_s/best.pt
train() { local name=${1%%|*} args=${1#*|}; [ -f outputs/$name/final.pt ] && return
  python scripts/e4b_train.py --kind motiondec --run $name $args --steps 15000 --workers 4 > outputs/$name.log 2>&1; }
phase() {
  local runs=("$@") specs=()
  for i in $(seq 0 3 $((${#runs[@]} - 1))); do
    for j in 0 1 2; do [ $((i+j)) -lt ${#runs[@]} ] && train "${runs[$((i+j))]}" & done; wait
  done
  echo "TRAIN DONE $(date -Is)"
  for r in "${runs[@]}"; do specs+=("${r%%|*}:1"); done
  bash scripts/remote/e4b_eval_queue.sh "${specs[@]}"
}
k5=(); all=()
for seed in 0 1 2; do
  k5+=("e6_md_droid_k5_s$seed|--k 5 --init $DROID --seed $seed")
  k5+=("e6_md_scratch_k5_s$seed|--k 5 --init scratch --seed $seed")
  k5+=("e6_md_droid_noaux_k5_s$seed|--k 5 --init $DROID --state-loss 0 --seed $seed")
  all+=("e6_md_droid_all_s$seed|--k 0 --init $DROID --seed $seed")
  all+=("e6_md_scratch_all_s$seed|--k 0 --init scratch --seed $seed")
done
phase "${k5[@]}"
phase "${all[@]}"
echo "E6 DONE $(date -Is)"
