#!/bin/bash
# Build/unpack the env on a GB10 node and smoke-test torch + an offscreen LIBERO render + a short TrunkBC train/eval.
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 03:00:00
#SBATCH -J lt-sopt-check
#SBATCH -o /media/cvpr/langtian/SOPT/logs/%x-%j.out
source /media/cvpr/langtian/SOPT/scripts_dgx/common.sh
echo "job $SLURM_JOB_ID node $(hostname) $(date -Is) commit $COMMIT"
setup_env || exit 1
setup_repo || exit 1
cd "$REPO"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device="cuda"); print("matmul ok", (x@x).abs().mean().item())' || exit 1
python - <<'PY' || exit 1
import os
from sopt.sim.libero_env import setup_env_vars, LiberoTaskEnv
setup_env_vars(os.environ["SOPT_DATA"])
e = LiberoTaskEnv("libero_90", 0)
print("libero render ok", type(e).__name__)
PY
r=dgxsmoke_trunk_droid
python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 200 --workers 8 --init outputs/e2_ar_flow_f100_s/best.pt \
  --seed 0 --tasks configs/libero90_17_tasks.json > outputs/$r.log 2>&1; echo "train exit $?"; tail -3 outputs/$r.log
python scripts/e4b_eval.py --policy outputs/$r/final.pt --idm outputs/e4b_idm/final.pt --n-init 2 \
  --out outputs/$r/rollouts_smoke.json > outputs/$r.eval.log 2>&1; echo "eval exit $?"; tail -3 outputs/$r.eval.log
sync_out $r; sync_frames
echo "done $(date -Is)"
