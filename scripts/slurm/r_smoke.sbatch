#!/bin/bash
# R2 smoke test before any expert_kin run (2026-11-16): (1) with the host weights of a trained `none` checkpoint, a fresh
# expert_kin policy (and expert_droid, as a reference) must reproduce `none` exactly and its injection must be live, on
# DP and SmolVLA (scripts/r_init_check.py); (2) a 30-step training run + 1-init
# evaluation of expert_kin on each host must finish, with the fitted kinematic constants stored in the checkpoint.
#   sbatch scripts/slurm/r_smoke.sbatch
#SBATCH --job-name=sopt-rsmoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/SOPT/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=/scratch/ll5582/SOPT_DATA/torch_cache
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
fail=0
for h in dp:e21_dp_none_k5_t17_s0 smolvla:e23_smolvla_none_k5_jr_t17_s0; do n=${h%%:*}; c=${h#*:}
  python scripts/r_init_check.py --ckpt outputs/$c/final.pt --out outputs/r_init_$n.json > logs/r_init_$n.log 2>&1
  echo "init check $n exit $? $(date -Is)"; tail -3 logs/r_init_$n.log
  grep -q "init check PASS" logs/r_init_$n.log || fail=1
done
for v in diffusion:64 smolvla:32; do n=${v%%:*}; bs=${v#*:}; r=r_smoke_${n}_kin
  rm -rf outputs/$r
  python scripts/vla_train.py --vla $n --arm expert_kin --k 5 --suite libero90_17 --seed 0 --run $r --steps 30 --bs $bs \
    --workers 8 > logs/$r.log 2>&1 && grep "kinematic features" logs/$r.log && \
  python scripts/vla_eval.py --ckpt outputs/$r/final.pt --out outputs/$r/rollouts.json --n-init 1 --max-steps 40 \
    > logs/$r.eval.log 2>&1
  rc=$?; echo "smoke $n exit $rc $(date -Is)"; [ $rc = 0 ] || { tail -5 logs/$r.log logs/$r.eval.log; fail=1; }
  python - <<PY || fail=1
import torch
ck = torch.load("outputs/$r/final.pt", map_location="cpu", weights_only=False)
kin = {k: v for k, v in ck["trainable"].items() if ".kin_" in k}
assert ck["arm"] == "expert_kin" and len(kin) == 3, sorted(kin)
ready = [v for k, v in kin.items() if k.endswith("kin_ready")][0]
std = [v for k, v in kin.items() if k.endswith("kin_std")][0]
assert ready.item() == 1.0 and std.numel() == 85 and (std > 0).all(), (ready, std.shape)
print("$n checkpoint carries fitted kinematic constants:", sorted(kin), "std", float(std.min()), float(std.max()))
PY
done
echo "SMOKE $([ $fail = 0 ] && echo PASS || echo FAIL) end=$(date -Is)"
