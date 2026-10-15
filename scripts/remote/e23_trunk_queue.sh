#!/usr/bin/env bash
# E23 on 127: TrunkBC protocol (E4b) on the 17-task LIBERO-90 set (configs/libero90_17_tasks.json): BC / scratch trunk /
# DROID trunk, k5, 3 seeds, 50 inits per task. Starts after E19 (same GPU).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
until grep -q "E19 DONE" outputs/e19_queue.log 2>/dev/null; do sleep 600; done
echo "start $(date -Is) $(git rev-parse --short HEAD)"
T=configs/libero90_17_tasks.json
for seed in 0 1 2; do
  for v in bc:bc:scratch trunk_scratch:trunkbc:scratch trunk_droid:trunkbc:outputs/e2_ar_flow_f100_s/best.pt; do
    IFS=: read name kind init <<< "$v"
    r=e23_${name}_k5_t17_s$seed
    [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind $kind --k 5 --steps 15000 --workers 4 \
      --init $init --seed $seed --tasks $T > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
    [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
      --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
      echo "$r eval exit $? $(date -Is)"; }
  done
done
echo "E23 TRUNK DONE $(date -Is)"
