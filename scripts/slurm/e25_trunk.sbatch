#!/bin/bash
# E25 / E33-TrunkBC on Jubail (moved from scripts/dgx/e25_pretrain.sbatch on 2026-10-23: DGX unreachable since 10-17,
# its e23 reference rollouts are not reachable either). One SOPT pretraining variant (VARIANT), then the E23 TrunkBC
# protocol with it (17 tasks, k5, seeds 0-2, 50 inits) and the E24 probes. Every arm, including the reference, is
# re-run here so all comparisons are same-cluster and same-code:
#   ref     no pretraining: the existing full-state ctx96 prior outputs/e2_ar_flow_f100_s (= e23_trunk_droid)
#   base    e2 recipe, pretraining seed=1 (prior-instance variance; E35 showed it can matter downstream)
#   speed   data.speed_aug=[0.5,2.0] data.speed_aug_prob=0.8
#   ctx48 / ctx192 / ctx384   history 3.2 / 12.8 / 25.6 s; left_pad = ctx - 96 as in E33 (same anchors as ctx96)
#   scratch random-init trunk (the E23 scratch arm)
# Pretraining = e2_ar_flow_f100_s recipe (S, ar_flow, 100% DROID, full state, 40k steps). DROID states only.
#   SBATCH_GRES=gpu:a100:1 VARIANT=ctx192 sbatch scripts/slurm/e25_trunk.sbatch
#SBATCH --job-name=sopt-e25
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
V=${VARIANT:?set VARIANT}
case $V in
  ref|scratch) OV="" ;;
  base) OV="seed=1" ;;
  speed) OV="data.speed_aug=[0.5,2.0] data.speed_aug_prob=0.8" ;;
  ctx48) OV="model.ctx_len=48" ;;
  ctx192) OV="model.ctx_len=192 data.left_pad=96" ;;
  ctx384) OV="model.ctx_len=384 data.left_pad=288" ;;
  *) echo "unknown VARIANT $V"; exit 1 ;;
esac
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) variant=$V ($OV) start=$(date -Is)"
[ -f outputs/e4b_idm/final.pt ] || { echo "no outputs/e4b_idm"; exit 1; }
case $V in
  ref) P=outputs/e2_ar_flow_f100_s; INIT=$P/best.pt; RP=e25j_trunk_ref ;;
  scratch) P=""; INIT=scratch; RP=e25j_trunk_scratch ;;
  *) P=outputs/e25j_$V; INIT=$P/best.pt; RP=e25j_trunk_$V
     [ -f $P/best.pt ] || { python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e2.yaml \
       model.objective=ar_flow data.root=$SOPT_DATA/processed/droid run_name=e25j_$V train.num_workers=8 $OV \
       > logs/e25j_$V.log 2>&1; echo "pretrain exit $? $(date -Is)"; tail -n 2 logs/e25j_$V.log; }
     [ -f $P/best.pt ] || exit 1 ;;
esac
T=configs/libero90_17_tasks.json
for seed in 0 1 2; do
  r=${RP}_k5_t17_s$seed
  [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 15000 --workers 8 \
    --init $INIT --seed $seed --tasks $T > logs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; }
  [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
    --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > logs/$r.eval_n50.log 2>&1
    echo "$r eval exit $? $(date -Is)"; }
done
[ -n "$P" ] && [ ! -f $P/probe.json ] && { python scripts/e24_probe.py --out $P/probe.json --no-rand --multi \
  --extra $V=$P/best.pt > logs/e25j_$V.probe.log 2>&1; echo "probe exit $? $(date -Is)"; tail -n 4 logs/e25j_$V.probe.log; }
echo "end=$(date -Is)"
