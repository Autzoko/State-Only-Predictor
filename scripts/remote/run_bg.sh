#!/usr/bin/env bash
# On 127 / 3090: run training detached in tmux; log -> outputs/<run_name>.log.
#   GPU=0 scripts/remote/run_bg.sh e1_ar_flow_s --config configs/model/s.yaml --config configs/experiment/e1_ar_flow.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?run_name}; shift
mkdir -p outputs
tmux new-session -d -s "sopt_$RUN" \
  "source scripts/remote/env.sh && CUDA_VISIBLE_DEVICES=${GPU:-0} python scripts/train.py \
   data.root=\$SOPT_DATA/processed/droid $* run_name=$RUN 2>&1 | tee outputs/$RUN.log"
echo "started tmux session sopt_$RUN"
