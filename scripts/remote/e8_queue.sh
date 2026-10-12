#!/usr/bin/env bash
# E8: does SOPT still help on top of a strong frozen pretrained vision encoder (DINOv2-S)?
# BC vs TrunkBC scratch vs TrunkBC DROID, all with --vision dinov2; k5 then all demos; 3 seeds; 50 init states.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DROID=outputs/e2_ar_flow_f100_s/best.pt
train() { local name=${1%%|*} args=${1#*|}; [ -f outputs/$name/final.pt ] && return
  python scripts/e4b_train.py --run $name $args --vision dinov2 --steps 15000 --workers 4 > outputs/$name.log 2>&1; }
evaluate() { [ -f outputs/$1/rollouts_n50.json ] && return
  python scripts/e4b_eval.py --policy outputs/$1/final.pt --idm outputs/e4b_idm/final.pt --n-init 50 \
    --out outputs/$1/rollouts_n50.json > outputs/$1.eval_n50.log 2>&1; echo "$1 eval exit $? $(date -Is)"; }
phase() {
  local runs=("$@")
  for i in $(seq 0 3 $((${#runs[@]} - 1))); do
    for j in 0 1 2; do [ $((i+j)) -lt ${#runs[@]} ] && train "${runs[$((i+j))]}" & done; wait
  done
  echo "TRAIN DONE $(date -Is)"
  for r in "${runs[@]}"; do evaluate "${r%%|*}"; done
}
for tag in k5 all; do
  k=$([ $tag = all ] && echo 0 || echo 5); runs=()
  for seed in 0 1 2; do
    runs+=("e8_dino_bc_${tag}_s$seed|--kind bc --k $k --seed $seed")
    runs+=("e8_dino_trunk_scratch_${tag}_s$seed|--kind trunkbc --k $k --init scratch --seed $seed")
    runs+=("e8_dino_trunk_droid_${tag}_s$seed|--kind trunkbc --k $k --init $DROID --seed $seed")
  done
  phase "${runs[@]}"
  echo "PHASE $tag DONE $(date -Is)"
done
echo "E8 DONE $(date -Is)"
