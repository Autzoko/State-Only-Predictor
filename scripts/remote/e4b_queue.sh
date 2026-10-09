#!/usr/bin/env bash
# E4b on one GPU: train 8 policies (4 concurrent), then closed-loop eval (2 concurrent). Run inside tmux.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DROID=outputs/e2_ar_flow_f100_s/best.pt
DROID_LIBERO=outputs/e4a_ar_flow_ft_kall/best.pt
runs=()
for k in 5 0; do
  tag=$([ $k = 0 ] && echo all || echo k$k)
  runs+=("e4b_prior_scratch_$tag|--kind prior --k $k --init scratch")
  runs+=("e4b_prior_droid_$tag|--kind prior --k $k --init $DROID")
  runs+=("e4b_prior_droidlibero_$tag|--kind prior --k $k --init $DROID_LIBERO")
  runs+=("e4b_bc_$tag|--kind bc --k $k")
done
train() { local name=${1%%|*} args=${1#*|}; [ -f outputs/$name/final.pt ] && return
  python scripts/e4b_train.py --run $name $args --steps 15000 --workers 5 > outputs/$name.log 2>&1; }
evaluate() { local name=${1%%|*}; [ -f outputs/$name/rollouts.json ] && return
  python scripts/e4b_eval.py --policy outputs/$name/final.pt --idm outputs/e4b_idm/final.pt --n-init 20 \
    --out outputs/$name/rollouts.json > outputs/$name.eval.log 2>&1; }
# Decode all demo videos once, single process (the k=5 subsets are contained in "all").
python - <<'PY'
import json, os
from pathlib import Path
from sopt.policy.data import Raw20, build_frame_cache, select_episodes
d = Path(os.environ["SOPT_DATA"]) / "processed"
raw = Raw20(d / "libero90_raw20")
tasks = json.loads(Path("configs/e4b_tasks.json").read_text())
sel = select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, None)
build_frame_cache(sel, d / "libero90_frames", workers=8)
print("frames cached:", len(sel), flush=True)
PY
for i in 0 4; do for j in 0 1 2 3; do train "${runs[$((i+j))]}" & done; wait; done
echo "TRAIN DONE $(date -Is)"
for i in 0 2 4 6; do evaluate "${runs[$i]}" & evaluate "${runs[$((i+1))]}" & wait; done
echo "ALL DONE $(date -Is)"
