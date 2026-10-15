#!/bin/bash
# SmolVLA (E23) or Diffusion Policy (E21) on the 17-task LIBERO-90 protocol on ONE DGX Spark node: the three arms of a
# seed run one after another on the same node (same hardware within a seed); recipe and run names identical to
# scripts/slurm/e23_smolvla.sbatch / e21_dp.sbatch. Finished runs are synced to $SHARE/outputs and skipped on resubmission.
#   bash scripts/dgx/submit.sh scripts/dgx/vla_host.sbatch --export=ALL,ENV_KIND=vla,HOST=smolvla,SEED=0
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 4-00:00:00
#SBATCH -J lt-sopt-vla
#SBATCH -o /media/cvpr/langtian/SOPT/logs/%x-%j.out
source /media/cvpr/langtian/SOPT/scripts_dgx/common.sh
HOST=${HOST:?smolvla|diffusion}; S=${SEED:-0}
case $HOST in smolvla) PFX=e23_smolvla STEPS=20000 BS=32 ;; diffusion) PFX=e21_dp STEPS=30000 BS=64 ;; *) exit 2 ;; esac
echo "job $SLURM_JOB_ID node $(hostname) $(date -Is) commit $COMMIT host=$HOST seed=$S"
setup_env || exit 1
setup_repo || exit 1
cd "$REPO"
rsync -r --ignore-existing "$SHARE/outputs/" outputs/ 2>/dev/null
smoke=${PFX}_dgxsmoke
if [ ! -f outputs/$smoke/rollouts.json ]; then
  python scripts/vla_train.py --vla $HOST --arm expert_droid --k 5 --suite libero90_17 --seed 0 --run $smoke --steps 30 \
    --bs $BS --workers 8 > outputs/$smoke.log 2>&1 && \
  python scripts/vla_eval.py --ckpt outputs/$smoke/final.pt --out outputs/$smoke/rollouts.json --n-init 1 --max-steps 40 \
    > outputs/$smoke.eval.log 2>&1
  rc=$?; echo "smoke exit $rc $(date -Is)"; tail -3 outputs/$smoke.log outputs/$smoke.eval.log; sync_out $smoke
  [ $rc = 0 ] || exit 1
fi
for arm in none expert_scratch expert_droid; do
  name=${PFX}_${arm}_k5_t17_s$S
  [ -f outputs/$name/final.pt ] || { python scripts/vla_train.py --vla $HOST --arm $arm --k 5 --suite libero90_17 \
    --seed $S --run $name --steps $STEPS --bs $BS --workers 8 > outputs/$name.log 2>&1; echo "$name train exit $? $(date -Is)"; }
  [ -f outputs/$name/rollouts_n50.json ] || { python scripts/vla_eval.py --ckpt outputs/$name/final.pt \
    --out outputs/$name/rollouts_n50.json --n-init 50 > outputs/$name.eval_n50.log 2>&1; echo "$name eval exit $? $(date -Is)"; }
  sync_out $name
done
echo "VLA HOST DONE $(date -Is)"
