#!/bin/bash
# R4 (reviewer item 3, 2026-11-18): does the host use the motion information in the history? Evaluation only, existing
# checkpoints: the history before the current frame is replaced at test time (src/sopt/policy/hist.py; static /
# reverse / swap), the current frame and everything else unchanged; 50 inits, clean scene, written next to the run's
# rollouts_n50.json (= `own`) as rollouts_n50_hist_<mode>.json. EVAL_PAR evaluations share the GPU, as cam_evals.
#   RUNS="e21_dp_expert_droid_k5_t17_s0 e21_dp_expert_kin_k5_t17_s0" sbatch scripts/slurm/r_hist.sbatch
#SBATCH --job-name=sopt-rhist
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/SOPT/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
MODES=${MODES:-static reverse swap}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) runs=${RUNS:?} modes=$MODES start=$(date -Is)"
for r in $RUNS; do [ -f outputs/$r/rollouts_n50.json ] || { echo "missing clean rollouts of $r"; exit 1; }; done
r0=${RUNS%% *}; mkdir -p outputs/r_smoke_hist  # smoke: every mode runs on the first checkpoint (1 init, 40 steps)
for m in $MODES; do
  python scripts/vla_eval.py --ckpt outputs/$r0/final.pt --n-init 1 --max-steps 40 --hist $m \
    --out outputs/r_smoke_hist/${r0}_$m.json > logs/r_smoke_hist_${r0}_$m.log 2>&1 || { echo "smoke $r0 $m FAILED"; tail -n 5 logs/r_smoke_hist_${r0}_$m.log; exit 1; }
done
echo "smoke PASS $(date -Is)"
todo=()
for r in $RUNS; do for m in $MODES; do [ -f outputs/$r/rollouts_n50_hist_$m.json ] || todo+=("$r:$m"); done; done
P=${EVAL_PAR:-2}
for ((i = 0; i < ${#todo[@]}; i += P)); do
  for x in "${todo[@]:i:P}"; do (r=${x%%:*}; m=${x#*:}; f=outputs/$r/rollouts_n50_hist_$m.json
    python scripts/vla_eval.py --ckpt outputs/$r/final.pt --n-init 50 --hist $m --out $f > logs/$r.hist_$m.log 2>&1
    echo "$r hist:$m exit $? $(date -Is) $(python -c "import json;d=json.load(open('$f'));print(d['hist'],d['n_episodes'],d['success_rate'])" 2>/dev/null)") &
  done; wait
done
echo "end=$(date -Is)"
