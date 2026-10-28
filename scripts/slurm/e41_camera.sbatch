#!/bin/bash
# E41 (pre-registered in docs/EXPERIMENTS.md): follow-ups of E37's camera result. Evaluation only, the E37 checkpoints
# and seeds (0, 1), 50 inits, perturbations from src/sopt/sim/perturb.py (camera:1/2/3 = agentview yaw 10/20/30 deg).
#   dose   none + SOPT arm at camera:1 and camera:3 (camera:2 and clean exist from E37 / the main table)
#   mask   SOPT arm with the SOPT input switched off for the whole episode (vla_eval --sopt-mask-grasp -1, as E34),
#          clean and camera:2 -- DP and GR00T only (SmolVLA computes its SOPT feature inside forward; no mask hook)
#   SBATCH_GRES=gpu:a100:1 HOST=dp|smolvla_goal|groot SEED=0|1 sbatch scripts/slurm/e41_camera.sbatch
#SBATCH --job-name=sopt-e41
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
H=${HOST:?}; S=${SEED:?}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) host_policy=$H seed=$S start=$(date -Is)"
case $H in  # same pairs as e37_robust.sbatch
  dp) NONE=e21_dp_none_k5_t17_s$S; SOPT=e21_dp_expert_droid_k5_t17_s$S ;;
  smolvla_goal) NONE=e13_libero_goal_none_k5_s$S; SOPT=e13_libero_goal_expert_droid_k5_s$S ;;
  groot) NONE=e22_groot_none_k5_t17_s$S; SOPT=e31_groot_xattn_droid_k5_t17_s$S ;;
  *) echo "unknown host $H"; exit 1 ;;
esac
ev() {  # $1 run, $2 perturb spec or "clean", $3 tag, extra args...
  local r=$1 p=$2 t=$3; shift 3; local o=outputs/$r/e41_${t}_${p/:/}
  [ -f $o.json ] || python scripts/vla_eval.py --ckpt outputs/$r/final.pt --n-init 50 \
    $([ $p = clean ] || echo --perturb $p) --out $o.json "$@" > logs/$r.e41_${t}_${p/:/}.log 2>&1
  echo "$r $p $t exit $? $(date -Is) $(python -c "import json;print(json.load(open('$o.json'))['success_rate'])" 2>/dev/null)"; }
todo=()
for r in $NONE $SOPT; do for p in camera:1 camera:3; do todo+=("$r $p dose"); done; done
if [ $H != smolvla_goal ]; then for p in clean camera:2; do todo+=("$SOPT $p mask --sopt-mask-grasp -1"); done; fi
# Evaluations are independent (own simulator processes, episodes seeded by init) and simulation-bound (~15% GPU when
# run one at a time), so EVAL_PAR of them share the GPU; peak RSS ~70 GB per evaluation, 2 fit in --mem=160G.
P=${EVAL_PAR:-2}
for ((i = 0; i < ${#todo[@]}; i += P)); do for x in "${todo[@]:i:P}"; do ev $x & done; wait; done
echo "end=$(date -Is)"
