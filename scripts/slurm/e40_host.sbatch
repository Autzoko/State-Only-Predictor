#!/bin/bash
# E40 (docs/EXPERIMENTS.md, pre-registered): does SOPT's gain shrink once the host itself has action pretraining?
# Stage 1 (this job): Diffusion Policy arm none pretrained on every demo of the LIBERO-90 *train-split* tasks (52 tasks;
# none of the 17 evaluated tasks, asserted in vla_train.py), one-hot over all LIBERO-90 tasks, 100k steps at batch 64.
# A 30-step smoke of the whole chain (pretrain -> fine-tune expert_droid from it -> 1-init eval) runs first.
# Stage 2: scripts/slurm/e21_dp.sbatch with INIT_HOST=outputs/e40_dp_host_train/final.pt TAG=_hpre (k5, 17 tasks).
#   SBATCH_GRES=gpu:a100:1 sbatch scripts/slurm/e40_host.sbatch
#SBATCH --job-name=sopt-e40
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=/scratch/ll5582/SOPT_DATA/torch_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
H=e40_dp_host_train; HSTEPS=${HSTEPS:-100000}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) hsteps=$HSTEPS start=$(date -Is)"
if [ ! -f outputs/e40_smoke_ft/rollouts.json ]; then
  python scripts/vla_train.py --vla diffusion --arm none --suite libero90_train --k 1 --task-vocab all --seed 0 \
    --run e40_smoke_host --steps 30 --bs 64 --workers 8 > logs/e40_smoke_host.log 2>&1 && \
  python scripts/vla_train.py --vla diffusion --arm expert_droid --suite libero90_17 --k 5 --task-vocab all --seed 0 \
    --init-host outputs/e40_smoke_host/final.pt --run e40_smoke_ft --steps 30 --bs 64 --workers 8 > logs/e40_smoke_ft.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/e40_smoke_ft/final.pt --out outputs/e40_smoke_ft/rollouts.json --n-init 1 \
    --max-steps 40 > logs/e40_smoke_ft.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; grep -h "init-host" logs/e40_smoke_ft.log
  [ $rc = 0 ] || exit 1
fi
[ -f outputs/$H/final.pt ] || { python scripts/vla_train.py --vla diffusion --arm none --suite libero90_train --k 0 \
  --task-vocab all --seed 0 --run $H --steps $HSTEPS --bs 64 --workers 8 > logs/$H.log 2>&1; echo "host pretrain exit $? $(date -Is)"; }
python -c "import json;m=json.load(open('outputs/$H/meta.json'));print('host episodes', len(m['episodes']), 'samples', m['samples'])"
echo "end=$(date -Is)"
