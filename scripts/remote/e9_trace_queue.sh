#!/usr/bin/env bash
# E9a failure taxonomy: re-run existing k5 policies (seed 0) with simulator-state tracing, 50 init states.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
for r in e4b_bc_k5 e4b_trunkbc_scratch_k5 e4b_trunkbc_droid_k5 e4b_bc_all e4b_trunkbc_droid_all; do
  [ -f outputs/$r/trace_n50.json ] && continue
  python scripts/e4b_eval.py --policy outputs/$r/final.pt --idm outputs/e4b_idm/final.pt --n-init 50 --trace 4 \
    --out outputs/$r/trace_n50.json > outputs/$r.trace_n50.log 2>&1; echo "$r trace exit $? $(date -Is)"
done
echo "E9A DONE $(date -Is)"
