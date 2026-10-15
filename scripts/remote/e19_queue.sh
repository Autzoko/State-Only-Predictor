#!/usr/bin/env bash
# E19 on 127 (A6000, SOPT_VLA_ENV with LeRobot 0.6.2): SmolVLA + DROID SOPT with SOPT dropout CDROP (default 0.5),
# LIBERO-90 E4b tasks k5, seeds 0-2 in sequence; identical to scripts/slurm/e19_cdrop.sbatch with SUITE=libero90.
# Paired with E7 (Jubail outputs) by scripts/agg_e19.py after copying the rollouts back.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/langtian/SOPT_VLA_ENV/bin:$PATH PYTHONPATH=$PWD/src SOPT_DATA=$HOME/langtian/SOPT_DATA
export HF_HOME=$SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=${GPU:-0}
CD=${CDROP:-0.5}; CDT=cd${CD/./}
echo "start $(date -Is) $(git rev-parse --short HEAD) cdrop=$CD"
for seed in 0 1 2; do
  r=e19_vla_expert_droid_${CDT}_k5_s$seed
  [ -f outputs/$r/final.pt ] || { python scripts/vla_train.py --arm expert_droid --k 5 --seed $seed --run $r --steps 20000 \
    --bs 32 --workers 6 --cond-drop $CD > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
  [ -f outputs/$r/rollouts_n50.json ] || { python scripts/vla_eval.py --ckpt outputs/$r/final.pt \
    --out outputs/$r/rollouts_n50.json --n-init 50 > outputs/$r.eval_n50.log 2>&1; echo "$r eval exit $? $(date -Is)"; }
done
echo "E19 DONE $(date -Is)"
