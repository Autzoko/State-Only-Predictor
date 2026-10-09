#!/usr/bin/env bash
# E4b v3: prior + IDM with the per-step-normalized prior (DROID-pretrained vs scratch), K=16 mean plans.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PS=outputs/e1c_ar_flow_perstep_s/best.pt
runs=()
for k in 5 0; do
  tag=$([ $k = 0 ] && echo all || echo k$k)
  runs+=("e4b_prior_ps_scratch_$tag|--kind prior --k $k --init scratch --prior-override model.per_step_norm=true")
  runs+=("e4b_prior_ps_droid_$tag|--kind prior --k $k --init $PS")
done
for r in "${runs[@]}"; do name=${r%%|*}; args=${r#*|}
  [ -f outputs/$name/final.pt ] || python scripts/e4b_train.py --run $name $args --steps 15000 --workers 4 > outputs/$name.log 2>&1 &
done; wait
echo "TRAIN DONE $(date -Is)"
for i in 0 2; do for j in 0 1; do name=${runs[$((i+j))]%%|*}
  [ -f outputs/$name/rollouts_k16.json ] || python scripts/e4b_eval.py --policy outputs/$name/final.pt \
    --idm outputs/e4b_idm/final.pt --n-init 20 --plan-samples 16 --out outputs/$name/rollouts_k16.json \
    > outputs/$name.eval.log 2>&1 &
done; wait; done
echo "ALL DONE $(date -Is)"
