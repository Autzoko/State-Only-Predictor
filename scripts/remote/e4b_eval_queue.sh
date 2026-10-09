#!/usr/bin/env bash
# Sequential closed-loop evals (one at a time: each eval holds ~10 GB of GPU memory across its 7 workers).
#   bash scripts/remote/e4b_eval_queue.sh <run>:<plan_samples> ...
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
for spec in "$@"; do
  name=${spec%%:*}; k=${spec#*:}; out=$([ "$k" = 1 ] && echo rollouts.json || echo rollouts_k$k.json)
  [ -f outputs/$name/$out ] && continue
  python scripts/e4b_eval.py --policy outputs/$name/final.pt --idm outputs/e4b_idm/final.pt --n-init 20 \
    --plan-samples $k --out outputs/$name/$out > outputs/$name.eval_${out%.json}.log 2>&1
  echo "$name $out exit $? $(date -Is)"
done
echo "ALL DONE $(date -Is)"
