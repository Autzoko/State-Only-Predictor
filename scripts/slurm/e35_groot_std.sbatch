#!/bin/bash
# E35: GR00T N1.7 on a standard long-horizon suite (default libero_10, k5) -- does a strong host gain from SOPT history
# where SmolVLA's history component did (E13 libero_10: scratch +3.2, p=0.011)? GR00T recipe of E22, EE-only SOPT
# (prior e13_ar_flow_f100_s_ee, joints hidden as in E13), 20k steps, 50 inits x 10 tasks, OpenVLA step limits.
#   SET=a  none | expert_scratch | expert_droid        (additive, = E13 arms on GR00T)
#   SET=b  xattn_scratch | xattn_droid                 (E31 interface, no regularizer: E34 showed dropout 0.3 makes the
#                                                      branch unused)
#   SET=hist expert_scratch | expert_droid | xattn_droid with an E33 history-length prior (CTX=48|192|384:
#            outputs/e33_ar_flow_f100_s_ee_ctx$CTX, same recipe, only ctx_len changed; the policy's history = the
#            prior's ctx_len). Run names get _ctx$CTX; paired with the ctx96 runs of SET=a / b (same seed / demos / inits).
#   SET=histx xattn_scratch with the CTX prior's config (control: long history without DROID pretraining; 1 GPU,
#            SBATCH_GRES=gpu:a100:1)
#   SET=key  none | xattn_droid (ctx96) | xattn_droid with the CTX prior: the decisive comparison for extra seeds
#            (same names as SET=a / b / hist, so seeds pool directly)
#   SET=one  a single arm ARM=<arm> on 1 GPU (SBATCH_GRES=gpu:a100:1; CTX=<n> for an E33 prior, unset = ctx96):
#            1-GPU jobs start within hours while 3-GPU jobs wait a day on the busy nvidia partition
#   SUITE=libero_10 SET=a SEED=0 sbatch scripts/slurm/e35_groot_std.sbatch
#   2-arm SETs must request 2 GPUs (SBATCH_GRES=gpu:a100:2 overrides the directive below): Jubail cancels jobs
#   holding an idle GPU (E35 SET=b was cancelled twice, at 04:00 and 08:00, with 1 of 3 GPUs unused).
#SBATCH --job-name=sopt-e35
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
SUITE=${SUITE:-libero_10}; S=${SEED:-0}; SET=${SET:-a}; K=${K:-5}; PRIOR=outputs/e13_ar_flow_f100_s_ee/best.pt; PT=""
[[ $SET = hist* || ( $SET = one && -n ${CTX:-} ) ]] && { PRIOR=outputs/e33_ar_flow_f100_s_ee_ctx${CTX:?set CTX}/best.pt; PT=_ctx$CTX; }
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) suite=$SUITE set=$SET seed=$S prior=$PRIOR start=$(date -Is)"
if [ ! -f outputs/e35_smoke_$SUITE/rollouts.json ]; then  # GR00T has not run on a standard suite before
  python scripts/vla_train.py --vla groot --arm xattn_droid --k $K --suite $SUITE --seed 0 --run e35_smoke_$SUITE \
    --prior $PRIOR --steps 30 --bs 32 --workers 8 > logs/e35_smoke.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e35_smoke_$SUITE/final.pt --out outputs/e35_smoke_$SUITE/rollouts.json \
    --n-init 1 --max-steps 40 > logs/e35_smoke.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; [ $rc = 0 ] || { tail -5 logs/e35_smoke.log logs/e35_smoke.eval.log; exit 1; }
fi
run() { local name=e35_groot_${SUITE}_$1_k${K}${PT}_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_train.py --vla groot --arm $1 --k $K \
    --suite $SUITE --seed $S --run $name --steps 20000 --bs 32 --workers 8 --prior $PRIOR > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$2 python scripts/vla_eval.py \
    --ckpt outputs/$name/final.pt --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"; }
case $SET in
  a) run none 0 & run expert_scratch 1 & run expert_droid 2 & wait ;;
  b) run xattn_scratch 0 & run xattn_droid 1 & wait ;;
  hist) run expert_scratch 0 & run expert_droid 1 & run xattn_droid 2 & wait ;;
  histx) run xattn_scratch 0 ;;
  one) run ${ARM:?set ARM} 0 ;;
  key) run none 0 & run xattn_droid 1 &
       (PRIOR=outputs/e33_ar_flow_f100_s_ee_ctx${CTX:?set CTX}/best.pt PT=_ctx$CTX; run xattn_droid 2) & wait ;;
  *) echo "unknown SET $SET"; exit 1 ;;
esac
echo "end=$(date -Is)"
