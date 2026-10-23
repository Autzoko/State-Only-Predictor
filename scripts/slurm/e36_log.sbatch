#!/bin/bash
# E36a data + analysis (docs/SOPT_NEXT.md): re-run the clean closed-loop evals of existing checkpoints with
# `--log-npz` (per-step joints + fingers, executed actions, every predicted chunk), check they reproduce the original
# rollouts_n50.json, then the host-agnostic failure-prediction analysis (scripts/e36_failure.py).
# Runs (seed 0, LIBERO-90 17 tasks, 50 inits; one host per family, the SOPT arm only where it is part of a host
# comparison already in the paper): GR00T none / xattn_droid, DP none / expert_droid, pi0.5 none, X-VLA none;
# long horizon (libero_10, seed 3 = pre-registered seed): GR00T none / xattn_droid ctx384.
# Outputs: outputs/<run>/e36_clean_n50.{json,npz}; outputs/e36_failure.json. Never overwrites rollouts_n50.json.
#   SBATCH_GRES=gpu:a100:1 sbatch scripts/slurm/e36_log.sbatch
#SBATCH --job-name=sopt-e36
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
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
declare -A RUNS=([groot_none]=e22_groot_none_k5_t17_s0 [groot_xattn]=e31_groot_xattn_droid_k5_t17_s0
  [dp_none]=e21_dp_none_k5_t17_s0 [dp_droid]=e21_dp_expert_droid_k5_t17_s0 [pi05_none]=e14_pi05_none_k5_lr1e-4_t17_s0
  [xvla_none]=e20_xvla_none_k5_t17_s0 [l10_groot_none]=e35_groot_libero_10_none_k5_s3
  [l10_groot_xattn384]=e35_groot_libero_10_xattn_droid_k5_ctx384_s3)
ORDER="groot_none groot_xattn dp_none dp_droid pi05_none xvla_none l10_groot_none l10_groot_xattn384"
ARGS=()
for k in $ORDER; do r=${RUNS[$k]}
  [ -f outputs/$r/e36_clean_n50.npz ] || python scripts/vla_eval.py --ckpt outputs/$r/final.pt --n-init 50 \
    --out outputs/$r/e36_clean_n50.json --log-npz outputs/$r/e36_clean_n50.npz > logs/$r.e36.log 2>&1
  echo "$r eval exit $? $(date -Is)"
  ARGS+=("$k=outputs/$r/e36_clean_n50.npz")
done
python - $ORDER <<'EOF'
import json, sys
runs = dict(groot_none="e22_groot_none_k5_t17_s0", groot_xattn="e31_groot_xattn_droid_k5_t17_s0",
            dp_none="e21_dp_none_k5_t17_s0", dp_droid="e21_dp_expert_droid_k5_t17_s0",
            pi05_none="e14_pi05_none_k5_lr1e-4_t17_s0", xvla_none="e20_xvla_none_k5_t17_s0",
            l10_groot_none="e35_groot_libero_10_none_k5_s3", l10_groot_xattn384="e35_groot_libero_10_xattn_droid_k5_ctx384_s3")
for k in sys.argv[1:]:
    r = runs[k]
    a = {(e["task"], e["init"]): e["success"] for e in json.load(open(f"outputs/{r}/e36_clean_n50.json"))["episodes"]}
    b = {(e["task"], e["init"]): e["success"] for e in json.load(open(f"outputs/{r}/rollouts_n50.json"))["episodes"]}
    ks = a.keys() & b.keys()
    print(f"{k:20s} logged {100*sum(a.values())/len(a):.1f} original {100*sum(b.values())/len(b):.1f} "
          f"episode agreement {sum(a[x] == b[x] for x in ks) / len(ks):.3f} (n {len(ks)})")
EOF
python scripts/e36_failure.py --out outputs/e36_failure.json --runs "${ARGS[@]}" \
  --primary groot_none groot_xattn dp_none dp_droid pi05_none xvla_none > logs/e36_failure.log 2>&1
echo "analysis exit $? $(date -Is)"; cat logs/e36_failure.log | cut -c1-400
echo "end=$(date -Is)"
