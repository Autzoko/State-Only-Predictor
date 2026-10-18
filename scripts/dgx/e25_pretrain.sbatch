#!/bin/bash
# E25 on one DGX Spark node: one SOPT pretraining variant (VARIANT env), then the E23 TrunkBC protocol with it
# (17 tasks, k5, seeds 0-2, 50 inits; paired against e23_trunk_droid_k5_t17_s*, i.e. e2_ar_flow_f100_s, run on DGX)
# and the E24 probes. Pretraining = the e2_ar_flow_f100_s recipe (S, ar_flow, 100% DROID, 40k steps) plus:
#   base    nothing (pretraining seed 1: the run-to-run noise of pretraining itself)
#   event   model.aux_event=0.5   (time to the next gripper event + where the fingers settle after it)
#   jepa    model.jepa=1.0        (latent prediction of the next H / P tokens of an EMA target encoder)
#   speed   data.speed_aug=[0.5,2.0] data.speed_aug_prob=0.8  (default 0.8-1.25, p 0.5)
#   ctx192  model.ctx_len=192     (12.8 s of history instead of 6.4 s; downstream uses the prior's ctx_len)
# E33 history-length ablation (with ctx192): ctx48 (3.2 s), ctx384 (25.6 s). Longer contexts use data.left_pad =
# ctx_len - 96, so every window still has >= 96 real context frames: the same episodes / anchors as ctx96, only older
# history added (padded with the first frame = robot at rest, exactly how LIBERO rollouts pad early history).
# Selection is pre-registered on the 7 dev tasks (scripts/agg_split.py); the 10 new tasks only confirm.
#   VARIANT=event bash scripts/dgx/submit.sh scripts/dgx/e25_pretrain.sbatch      (from dgx-login)
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 2-00:00:00
#SBATCH -J lt-sopt-e25
#SBATCH -o /media/cvpr/langtian/SOPT/logs/%x-%j.out
source /media/cvpr/langtian/SOPT/scripts_dgx/common.sh
V=${VARIANT:?set VARIANT}
case $V in
  base) OV="seed=1" ;;
  event) OV="model.aux_event=0.5" ;;
  jepa) OV="model.jepa=1.0" ;;
  speed) OV="data.speed_aug=[0.5,2.0] data.speed_aug_prob=0.8" ;;
  ctx192) OV="model.ctx_len=192 data.left_pad=96" ;;
  ctx48) OV="model.ctx_len=48" ;;
  ctx384) OV="model.ctx_len=384 data.left_pad=288" ;;
  *) echo "unknown VARIANT $V"; exit 1 ;;
esac
echo "job $SLURM_JOB_ID node $(hostname) $(date -Is) commit $COMMIT variant $V ($OV)"
setup_env || exit 1
setup_repo || exit 1
cd "$REPO"
[[ -f $SOPT_DATA/processed/droid/episodes.parquet ]] || { echo "no DROID states in $SOPT_DATA/processed/droid"; exit 1; }
rsync -r --ignore-existing "$SHARE/outputs/" outputs/ 2>/dev/null  # resume
P=outputs/e25_$V
if [[ ! -f $P/best.pt || ! -f $P/.done ]]; then
  python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e2.yaml model.objective=ar_flow \
    data.root=$SOPT_DATA/processed/droid run_name=e25_$V $OV > $P.log 2>&1
  rc=$?; echo "pretrain exit $rc $(date -Is)"; [[ $rc = 0 ]] || exit 1
  touch $P/.done
fi
sync_out e25_$V
T=configs/libero90_17_tasks.json
for seed in 0 1 2; do
  r=e25_trunk_${V}_k5_t17_s$seed
  [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind trunkbc --k 5 --steps 15000 --workers 8 \
    --init $P/best.pt --seed $seed --tasks $T > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; sync_frames; }
  [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
    --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
    echo "$r eval exit $? $(date -Is)"; }
  sync_out $r
done
python scripts/e24_probe.py --out $P/probe.json --no-rand --multi --extra $V=$P/best.pt > $P.probe.log 2>&1
echo "probe exit $? $(date -Is)"; tail -4 $P.probe.log
sync_out e25_$V
echo "E25 $V DONE $(date -Is)"
