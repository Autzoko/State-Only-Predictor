#!/bin/bash
# E23 on one DGX Spark node: TrunkBC protocol (E4b) on the 17-task LIBERO-90 set (configs/libero90_17_tasks.json):
# BC / scratch trunk / DROID trunk, k5, seeds 0-2, 50 inits per task -- same commands as scripts/remote/e23_trunk_queue.sh.
# Runs are synced to $SHARE/outputs as they finish; finished runs are skipped on resubmission.
#SBATCH -p spark
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 3-00:00:00
#SBATCH -J lt-sopt-e23
#SBATCH -o /media/cvpr/langtian/SOPT/logs/%x-%j.out
source /media/cvpr/langtian/SOPT/scripts_dgx/common.sh
echo "job $SLURM_JOB_ID node $(hostname) $(date -Is) commit $COMMIT"
setup_env || exit 1
setup_repo || exit 1
cd "$REPO"
rsync -r --ignore-existing "$SHARE/outputs/" outputs/ 2>/dev/null  # resume: runs finished on other nodes
T=configs/libero90_17_tasks.json
for seed in ${SEEDS:-0 1 2}; do
  for v in bc:bc:scratch trunk_scratch:trunkbc:scratch trunk_droid:trunkbc:outputs/e2_ar_flow_f100_s/best.pt; do
    IFS=: read name kind init <<< "$v"
    r=e23_${name}_k5_t17_s$seed
    [ -f outputs/$r/final.pt ] || { python scripts/e4b_train.py --run $r --kind $kind --k 5 --steps 15000 --workers 8 \
      --init $init --seed $seed --tasks $T > outputs/$r.log 2>&1; echo "$r train exit $? $(date -Is)"; sync_frames; }
    [ -f outputs/$r/rollouts_n50.json ] || { python scripts/e4b_eval.py --policy outputs/$r/final.pt \
      --idm outputs/e4b_idm/final.pt --n-init 50 --out outputs/$r/rollouts_n50.json > outputs/$r.eval_n50.log 2>&1
      echo "$r eval exit $? $(date -Is)"; }
    sync_out $r
  done
done
echo "E23 TRUNK DONE $(date -Is)"
