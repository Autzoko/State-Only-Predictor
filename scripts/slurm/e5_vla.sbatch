#!/bin/bash
# E5 on Jubail: SmolVLA (none / scratch / DROID motion tokens) x (k=5, all), one GPU per arm, then closed-loop eval.
# Uses the RoboticsNAS python env read-only (same lerobot commit 3f2c29e + hf-libero); code via PYTHONPATH.
#   sbatch scripts/slurm/e5_vla.sbatch
#SBATCH --job-name=sopt-e5-vla
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:3
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=30
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"; nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/prepare_libero_raw20.py --sopt-data $SOPT_DATA || exit 1
python - <<'PY' || exit 1
import json, os
from pathlib import Path
from sopt.policy.data import Raw20, build_frame_cache, select_episodes
d = Path(os.environ["SOPT_DATA"]) / "processed"
raw = Raw20(d / "libero90_raw20")
tasks = json.loads(Path("configs/e4b_tasks.json").read_text())
sel = select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, None)
build_frame_cache(sel, d / "libero90_frames256", workers=16, size=256)
print("frames cached:", len(sel), flush=True)
PY
for k in 5 0; do
  tag=$([ $k = 0 ] && echo all || echo k$k)
  gpu=0
  for arm in none scratch droid; do
    name=e5_vla_${arm}_$tag
    [ -f outputs/$name/final.pt ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_train.py --arm $arm --k $k --run $name \
      --steps 20000 --bs 32 --workers 8 > logs/$name.log 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  echo "trained $tag $(date -Is)"
done
for tag in k5 all; do
  gpu=0
  for arm in none scratch droid; do
    name=e5_vla_${arm}_$tag
    [ -f outputs/$name/rollouts.json ] || CUDA_VISIBLE_DEVICES=$gpu python scripts/vla_eval.py \
      --ckpt outputs/$name/final.pt --out outputs/$name/rollouts.json > logs/$name.eval.log 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  echo "evaluated $tag $(date -Is)"
done
echo "end=$(date -Is)"
