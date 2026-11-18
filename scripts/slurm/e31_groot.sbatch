#!/bin/bash
# E31: SOPT interfaces for GR00T N1.7 (docs/HOST_ANALYSIS.md section 10), 17-task LIBERO-90 protocol, k5, recipe of E22
# (paired with e22_groot_{none,expert_droid}_k5_t17_s<seed>: same seed, demos, init states).
#   SET=screen  adaln_droid | xattn_droid + modality dropout 0.3 + SOPT lr x0.1 | xattn_droid plain
#   SET=scratch adaln_scratch | xattn_scratch (+ the SET=screen regularizers) | (idle)   [controls, after screening]
#   SET=xscratch R1 (2026-11-16): xattn_scratch plain (no regularizers), the same-structure control of the plain
#               xattn_droid arm used in E37 / E41; one GPU; CAM=1 adds camera 10 / 20 / 30 deg evaluations
#   SET=xkin    R2 for GR00T (2026-11-18, reviewer item 1): xattn_kin = the plain xattn branch fed with explicit
#               kinematics; one GPU; CAM=1 adds the camera evaluations. Before training: zero-init + liveness check
#               against e22 none s0 (scripts/r_init_check.py) and a 30-step smoke run + 1-init eval, else abort.
#   SET=history E33 history length on the host: expert_droid | xattn_droid (+ regularizers) with a longer-context prior
#               (PRIOR=<ckpt> PTAG=_ctx192 ...; the policy's history = the prior's ctx_len, paired with the ctx96 runs)
# First: a fresh adaln / xattn policy must equal `none` exactly (zero init) and a 30-step smoke run + 1-init eval of
# each new arm must pass, else the job aborts.
#   SET=screen SEED=0 sbatch scripts/slurm/e31_groot.sbatch
#   2-arm SETs must request 2 GPUs (SBATCH_GRES=gpu:a100:2 overrides the directive below): Jubail cancels jobs
#   holding an idle GPU (E35 SET=b was cancelled twice, at 04:00 and 08:00, with 1 of 3 GPUs unused).
#SBATCH --job-name=sopt-e31
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/SOPT/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
S=${SEED:-0}; SET=${SET:-screen}; STEPS=${STEPS:-20000}; PRIOR=${PRIOR:-outputs/e2_ar_flow_f100_s/best.pt}; PT=${PTAG:-}
REG="--cond-drop 0.3 --prior-lr-mult 0.1"
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) set=$SET seed=$S prior=$PRIOR start=$(date -Is)"
if [ ! -f outputs/e31_init.json ]; then
  python scripts/vla_diag.py --out outputs/e31_init.json --n 16 --init-check --init-arms expert_droid adaln_droid xattn_droid \
    --ckpts outputs/e22_groot_none_k5_t17_s0/final.pt > logs/e31_init.log 2>&1
  echo "init check exit $? $(date -Is)"; grep "init check" logs/e31_init.log
  python - <<'EOF' || exit 1
import json; d = json.load(open("outputs/e31_init.json"))["init_check"]["max_abs_diff_none_vs_fresh"]
assert all(v == 0.0 for v in d.values()), d
EOF
fi
for a in adaln_droid xattn_droid; do
  [ -f outputs/e31_smoke_$a/rollouts.json ] && continue
  python scripts/vla_train.py --vla groot --arm $a --k 5 --suite libero90_17 --seed 0 --run e31_smoke_$a --steps 30 --bs 32 \
    --workers 8 $REG > logs/e31_smoke_$a.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e31_smoke_$a/final.pt --out outputs/e31_smoke_$a/rollouts.json --n-init 1 \
    --max-steps 40 > logs/e31_smoke_$a.eval.log 2>&1
  rc=$?; echo "smoke $a exit $rc $(date -Is)"; [ $rc = 0 ] || { tail -5 logs/e31_smoke_$a.log logs/e31_smoke_$a.eval.log; exit 1; }
done
run() { local arm=$1 gpu=$2 extra=$3 tag=$4 name=e31_groot_$1_k5$4${PT}_t17_s$S
  [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_train.py --vla groot --arm $arm --k 5 \
    --suite libero90_17 --seed $S --run $name --steps $STEPS --bs 32 --workers 8 --prior $PRIOR $extra > logs/$name.log 2>&1
  echo "$name train exit $? $(date -Is)"
  [ -f outputs/$name/rollouts_n50.json ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_eval.py --ckpt outputs/$name/final.pt \
    --out outputs/$name/rollouts_n50.json --n-init 50 > logs/$name.eval_n50.log 2>&1
  echo "$name eval exit $? $(date -Is)"
  [ -z "${CAM:-}" ] || cam_evals $name $gpu; }
source scripts/slurm/cam_evals.sh
if [ "$SET" = xkin ]; then
  if ! grep -q '"pass": true' outputs/r_init_groot_s$S.json 2>/dev/null; then
    python scripts/r_init_check.py --ckpt outputs/e22_groot_none_k5_t17_s0/final.pt --arms xattn_kin xattn_droid \
      --out outputs/r_init_groot_s$S.json > logs/r_init_groot_s$S.log 2>&1
    echo "init check groot exit $? $(date -Is)"; tail -n 3 logs/r_init_groot_s$S.log
    grep -q "init check PASS" logs/r_init_groot_s$S.log || exit 1
  fi
  r=r_smoke_groot_kin_s$S  # per job: two seeds may run at the same time
  if [ ! -f outputs/$r/rollouts.json ]; then
    python scripts/vla_train.py --vla groot --arm xattn_kin --k 5 --suite libero90_17 --seed 0 --run $r --steps 30 --bs 32 \
      --workers 8 > logs/$r.log 2>&1 && grep "kinematic features" logs/$r.log && \
    python scripts/vla_eval.py --ckpt outputs/$r/final.pt --out outputs/$r/rollouts.json --n-init 1 --max-steps 40 \
      > logs/$r.eval.log 2>&1
    rc=$?; echo "smoke groot kin exit $rc $(date -Is)"; [ $rc = 0 ] || { tail -n 5 logs/$r.log logs/$r.eval.log; exit 1; }
  fi
fi
case $SET in
  screen) run adaln_droid 0 "" "" & run xattn_droid 1 "$REG" _cd03_plr01 & run xattn_droid 2 "" "" & wait ;;
  scratch) run adaln_scratch 0 "" "" & run xattn_scratch 1 "$REG" _cd03_plr01 & wait ;;
  xscratch) run xattn_scratch 0 "" "" ;;
  xkin) run xattn_kin 0 "" "" ;;
  history) run expert_droid 0 "" "" & run xattn_droid 1 "$REG" _cd03_plr01 & wait ;;
  *) echo "unknown SET $SET"; exit 1 ;;
esac
echo "end=$(date -Is)"
