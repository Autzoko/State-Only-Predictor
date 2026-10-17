#!/bin/bash
# E26 + E27: why do GR00T / pi0.5 / X-VLA not gain from SOPT? (docs/HOST_ANALYSIS.md)
#  E26  re-run the closed-loop eval of the 17-task seed-0 checkpoints (X-VLA, GR00T, pi0.5 lr1e-4, DP as the positive
#       control; arms none / expert_scratch / expert_droid) with simulator traces + per-step action logs, then the
#       failure taxonomy and paired "which failures does SOPT create / remove" analysis (scripts/e26_taxonomy.py).
#       The traced success is checked against the original rollouts_n50.json (same seeds -> should reproduce).
#  E27  linear probes of the hosts' own action-decoder input for the E24 temporal targets (scripts/e27_host_probe.py).
# 3 GPUs: one arm per GPU (4 evals each, ~0.3-0.65 h per eval), then E27 on one GPU.
#   sbatch scripts/slurm/e26_trace.sbatch
#SBATCH --job-name=sopt-e26
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
RUNS="e20_xvla_%s_k5_t17_s0 e22_groot_%s_k5_t17_s0 e21_dp_%s_k5_t17_s0 e14_pi05_%s_k5_lr1e-4_t17_s0"
trace() { local a=$1 g=$2 h r
  for h in $RUNS; do r=$(printf $h $a)
    [ -f outputs/$r/trace_n50.json ] || CUDA_VISIBLE_DEVICES=$g python scripts/vla_eval.py --ckpt outputs/$r/final.pt \
      --out outputs/$r/trace_n50.json --n-init 50 --trace 4 --act-log > logs/$r.trace.log 2>&1
    echo "$r trace exit $? $(date -Is)"
  done; }
trace none 0 & trace expert_scratch 1 & trace expert_droid 2 & wait
python - <<'EOF'
import json, glob, os
for t in sorted(glob.glob("outputs/*_k5*_t17_s0/trace_n50.json")):
    r = t.replace("trace_n50.json", "rollouts_n50.json")
    if not os.path.exists(r): continue
    a = {(e["task"], e["init"]): e["success"] for e in json.load(open(t))["episodes"]}
    b = {(e["task"], e["init"]): e["success"] for e in json.load(open(r))["episodes"]}
    k = a.keys() & b.keys()
    print(f"{t.split('/')[1]:42s} traced {100*sum(a.values())/len(a):.1f} orig {100*sum(b.values())/len(b):.1f} "
          f"episode agreement {sum(a[x]==b[x] for x in k)/len(k):.3f}")
EOF
python scripts/e26_taxonomy.py --out outputs/e26_taxonomy.json outputs/{e20_xvla,e22_groot,e21_dp,e14_pi05}_*_k5*_t17_s0/trace_n50.json \
  > logs/e26_taxonomy.log 2>&1; echo "taxonomy exit $? $(date -Is)"; cat logs/e26_taxonomy.log | grep -v "^   "
C=()
for h in $RUNS; do for a in none expert_scratch expert_droid; do C+=(outputs/$(printf $h $a)/final.pt); done; done
CUDA_VISIBLE_DEVICES=0 python scripts/e27_host_probe.py --out outputs/e27_host_probe.json --ckpts "${C[@]}" > logs/e27_host_probe.log 2>&1
echo "e27 exit $? end=$(date -Is)"; grep -E "windows|^cur|^sopt|^host" logs/e27_host_probe.log
