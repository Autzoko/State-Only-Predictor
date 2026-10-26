#!/bin/bash
# E37: does SOPT make hosts more robust to visual distribution shift? (docs/SOPT_NEXT.md; perturbations in
# src/sopt/sim/perturb.py, checked by scripts/slurm/e37_probe.sbatch). Evaluation only, existing checkpoints.
#   MODE=pilot  level calibration on the seed-2 *none* checkpoints of the three hosts (never used in the test, and
#               no SOPT arm is looked at): every (kind, level), 10 inits. Rule fixed in advance: per kind, the lowest
#               level whose mean over hosts of (perturbed / clean success on the same inits) is <= 0.85, else level 3
#               -> outputs/e37_levels.json.
#   MODE=main HOST=dp|smolvla_goal|groot SEED=0|1 [ARM=none|droid|scratch]   both arms (or one) x 5 kinds at the chosen
#               level, 50 inits, --log-npz (GR00T: one arm per job, ~0.7 h per eval).
# Arms (the comparisons already reported clean): DP none vs expert_droid (17 tasks), SmolVLA libero_goal none vs
# expert_droid, GR00T none (E22) vs xattn_droid (E31).
# ARM=scratch (E37s, preregistered 2026-10-26): the same-structure random-init SOPT arm, DP e21_dp_expert_scratch and
# SmolVLA-goal e13_libero_goal_expert_scratch only (no 17-task GR00T xattn_scratch checkpoint exists).
#   SBATCH_GRES=gpu:a100:1 MODE=pilot sbatch scripts/slurm/e37_robust.sbatch
#SBATCH --job-name=sopt-e37
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
MODE=${MODE:?pilot|main}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) mode=$MODE host_policy=${HOST:-} seed=${SEED:-} start=$(date -Is)"
runs() {  # $1 host, $2 seed -> "none_run droid_run"
  case $1 in
    dp) echo "e21_dp_none_k5_t17_s$2 e21_dp_expert_droid_k5_t17_s$2" ;;
    smolvla_goal) echo "e13_libero_goal_none_k5_s$2 e13_libero_goal_expert_droid_k5_s$2" ;;
    groot) echo "e22_groot_none_k5_t17_s$2 e31_groot_xattn_droid_k5_t17_s$2" ;;
    *) echo "unknown host $1" >&2; exit 1 ;;
  esac; }
ev() {  # $1 run, $2 kind:level, $3 n_init, $4 tag
  local o=outputs/$1/e37_${4}_${2/:/}
  [ -f $o.json ] || python scripts/vla_eval.py --ckpt outputs/$1/final.pt --n-init $3 --perturb $2 --out $o.json \
    ${LOG:+--log-npz $o.npz} > logs/$1.e37_${4}_${2/:/}.log 2>&1
  echo "$1 $2 exit $? $(date -Is) $(python -c "import json;print(json.load(open('$o.json'))['success_rate'])" 2>/dev/null)"; }
if [ $MODE = pilot ]; then
  for h in dp smolvla_goal groot; do r=$(runs $h 2 | cut -d' ' -f1)
    for k in camera light texture noise robot; do for l in 1 2 3; do ev $r $k:$l 10 pilot; done; done
  done
  python - <<'EOF'
import json
runs = {"dp": "e21_dp_none_k5_t17_s2", "smolvla_goal": "e13_libero_goal_none_k5_s2", "groot": "e22_groot_none_k5_t17_s2"}
clean = {}
for h, r in runs.items():  # clean success of the same checkpoint on the same 10 inits (original rollouts)
    eps = [e for e in json.load(open(f"outputs/{r}/rollouts_n50.json"))["episodes"] if e["init"] < 10]
    clean[h] = sum(e["success"] for e in eps) / len(eps)
levels, table = {}, {}
for k in ("camera", "light", "texture", "noise", "robot"):
    for l in (1, 2, 3):
        rel = {h: json.load(open(f"outputs/{r}/e37_pilot_{k}{l}.json"))["success_rate"] / clean[h] for h, r in runs.items()}
        table[f"{k}:{l}"] = rel
        print(f"{k}:{l} relative success " + " ".join(f"{h} {v:.2f}" for h, v in rel.items()) + f" mean {sum(rel.values())/3:.2f}")
        if k not in levels and sum(rel.values()) / 3 <= 0.85:
            levels[k] = l
    levels.setdefault(k, 3)
json.dump({"levels": levels, "clean_10init": clean, "relative": table}, open("outputs/e37_levels.json", "w"), indent=1)
print("LEVELS", levels)
EOF
else
  H=${HOST:?}; S=${SEED:?}
  [ -f outputs/e37_levels.json ] || { echo "no outputs/e37_levels.json (run MODE=pilot first)"; exit 1; }
  set -- $(runs $H $S)
  case ${ARM:-both} in none) R=$1 ;; droid) R=$2 ;; both) R="$1 $2" ;;
    scratch) case $H in dp) R=e21_dp_expert_scratch_k5_t17_s$S ;; smolvla_goal) R=e13_libero_goal_expert_scratch_k5_s$S ;;
             *) echo "no scratch arm for $H"; exit 1 ;; esac
             [ -f outputs/$R/final.pt ] && [ -f outputs/$R/rollouts_n50.json ] || { echo "missing $R"; exit 1; } ;;
    *) echo "bad ARM"; exit 1 ;; esac
  for r in $R; do
    for k in camera light texture noise robot; do
      l=$(python -c "import json;print(json.load(open('outputs/e37_levels.json'))['levels']['$k'])")
      LOG=1 ev $r $k:$l 50 main
    done
  done
fi
echo "end=$(date -Is)"
