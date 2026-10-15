#!/usr/bin/env bash
# E18: where should state-only pretraining data come from? TrunkBC k5 x 3 seeds, 50 inits (E4b protocol), trunk init:
#   libero_only  e4a_ar_flow_scratch_kall: SOPT trained only on LIBERO-90 *train-split* states (52 other tasks,
#                2,725 demos; val-selected); in-domain robot / scenes, ~30x less data than DROID
#   droid_libero e4a_ar_flow_ft_kall: DROID prior post-trained on the same LIBERO-90 train-split states
# The E4b test tasks (configs/e4b_tasks.json) are in the test split, never seen by either. Compared with scratch /
# DROID-only (E4b) by scripts/agg_e18.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "start $(date -Is) $(git rev-parse --short HEAD)"
for seed in 0 1 2; do
  for v in libero_only:e4a_ar_flow_scratch_kall droid_libero:e4a_ar_flow_ft_kall; do
    r=e18_trunk_${v%%:*}_k5_s$seed
    [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 15000 --workers 4 \
      --init outputs/${v#*:}/best.pt --seed $seed > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
    [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
      --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
      echo "$r eval exit $? $(date -Is)"; }
  done
done
echo "E18 DONE $(date -Is)"
