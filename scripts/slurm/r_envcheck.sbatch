#!/bin/bash
# Environment-consistency check (2026-11-16): the R1-R3 arms run in the rebuilt env /scratch/ll5582/SOPT/env
# (build_env.sbatch) while the arms they are compared with ran in the former RoboticsNAS env. Re-run the clean 850-episode
# evaluation of existing checkpoints in the new env, written next to the original as rollouts_n50_envcheck.json, and
# compare per episode (paired McNemar) with the original rollouts_n50.json.
#   sbatch scripts/slurm/r_envcheck.sbatch      (RUNS overrides the checkpoints)
#SBATCH --job-name=sopt-renv
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/SOPT/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=/scratch/ll5582/SOPT_DATA/torch_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
RUNS=${RUNS:-e21_dp_none_k5_t17_s0 e23_smolvla_none_k5_jr_t17_s0}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) runs=$RUNS start=$(date -Is)"
for r in $RUNS; do (
  [ -f outputs/$r/rollouts_n50.json ] || { echo "missing outputs/$r/rollouts_n50.json"; exit 1; }
  python scripts/vla_eval.py --ckpt outputs/$r/final.pt --out outputs/$r/rollouts_n50_envcheck.json --n-init 50 \
    > logs/$r.eval_n50_envcheck.log 2>&1
  echo "$r envcheck exit $? $(date -Is)"
  python - <<PY
import json
a, b = (json.load(open(f"outputs/$r/{f}.json")) for f in ("rollouts_n50", "rollouts_n50_envcheck"))
print("$r old", a["success_rate"], "new", b["success_rate"], "episodes", a["n_episodes"], b["n_episodes"])
PY
  ) & done; wait
echo "end=$(date -Is)"
