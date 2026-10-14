#!/usr/bin/env bash
# E17: LM-style block (SwiGLU + RMSNorm + xavier-normal / scaled residual init) vs the original S block.
# Same recipe as e2_ar_flow_f100_s (S, ar_flow, 100% DROID, 40k steps), then TrunkBC k5 x 3 seeds, 50 inits
# (identical to E4b / E11; compared against e4b_trunkbc_droid_k5* by scripts/agg_e17.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/remote/env.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "start $(date -Is) $(git rev-parse --short HEAD)"
P=outputs/e17_ar_flow_f100_s_lm
if [ ! -f $P/best.pt ]; then
  python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e2.yaml model.objective=ar_flow \
    model.mlp=swiglu model.norm=rmsnorm model.init=xavier data.root=/home/nyuair/langtian/SOPT_DATA/processed/droid \
    run_name=e17_ar_flow_f100_s_lm > $P.log 2>&1
  echo "pretrain exit $? $(date -Is)"
fi
for seed in 0 1 2; do
  r=e17_trunk_lm_k5_s$seed
  [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 15000 --workers 4 \
    --init $P/best.pt --seed $seed > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
  [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
    --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
    echo "$r eval exit $? $(date -Is)"; }
done
echo "E17 DONE $(date -Is)"
