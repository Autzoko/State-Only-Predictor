#!/usr/bin/env bash
# E15b (127): E4b TrunkBC k5 with the EE-only DROID prior (joints hidden inside SOPT), 3 seeds, 50 inits.
# Paired controls: e4b_trunkbc_droid_k5* (joint prior) and e4b_bc_k5* / e4b_trunkbc_scratch_k5* (no prior).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "start $(date -Is)"
for seed in 0 1 2; do
  r=e15b_trunk_ee_k5_s$seed
  [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 15000 --workers 4 \
    --init outputs/e13_ar_flow_f100_s_ee/best.pt --seed $seed > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
  [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
    --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
    echo "$r eval exit $? $(date -Is)"; }
done
echo "E15b DONE $(date -Is)"
