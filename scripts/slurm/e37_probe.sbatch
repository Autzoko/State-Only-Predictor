#!/bin/bash
# E36 / E37 pre-flight on one GPU (nothing here is a result):
#  1. scripts/e37_probe.py: perturbation hooks on all 17 LIBERO-90 tasks + 2 libero_goal + 3 libero_10 scenes
#     (unperturbed path identical, determinism, hard reset restores the scene, images for inspection);
#  2. vla_eval.py refactor check: DP none s0, 2 inits, no perturbation, with --log-npz -> episodes must match the
#     original rollouts_n50.json, and the npz offsets / lengths must match the episode steps;
#  3. smoke of a perturbed eval (camera:2, 1 init, 40 steps) on DP and on GR00T xattn_droid.
#   SBATCH_GRES=gpu:a100:1 sbatch scripts/slurm/e37_probe.sbatch
#SBATCH --job-name=sopt-e37p
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/e37_probe.py --out outputs/e37_probe > logs/e37_probe.log 2>&1
echo "probe exit $? $(date -Is)"; tail -n 30 logs/e37_probe.log
D=outputs/e37_check; mkdir -p $D
python scripts/vla_eval.py --ckpt outputs/e21_dp_none_k5_t17_s0/final.pt --out $D/dp_none_clean.json --n-init 2 \
  --log-npz $D/dp_none_clean.npz > logs/e37_check_clean.log 2>&1; echo "clean eval exit $? $(date -Is)"
python scripts/vla_eval.py --ckpt outputs/e21_dp_none_k5_t17_s0/final.pt --out $D/dp_none_cam2.json --n-init 1 \
  --max-steps 40 --perturb camera:2 --log-npz $D/dp_none_cam2.npz > logs/e37_check_cam.log 2>&1; echo "dp cam eval exit $?"
python scripts/vla_eval.py --ckpt outputs/e31_groot_xattn_droid_k5_t17_s0/final.pt --out $D/groot_x_noise2.json --n-init 1 \
  --max-steps 40 --perturb noise:2 --log-npz $D/groot_x_noise2.npz > logs/e37_check_groot.log 2>&1; echo "groot noise eval exit $?"
python - <<'EOF'
import json, numpy as np
D = "outputs/e37_check"
new = json.load(open(f"{D}/dp_none_clean.json"))["episodes"]
old = {(e["task"], e["init"]): e for e in json.load(open("outputs/e21_dp_none_k5_t17_s0/rollouts_n50.json"))["episodes"]}
agree = [e["success"] == old[(e["task"], e["init"])]["success"] for e in new]
same_steps = [e["steps"] == old[(e["task"], e["init"])]["steps"] for e in new]
print(f"clean re-eval vs original: success agreement {np.mean(agree):.3f}, identical steps {np.mean(same_steps):.3f} over {len(new)} episodes")
for name in ("dp_none_clean", "dp_none_cam2", "groot_x_noise2"):
    js = json.load(open(f"{D}/{name}.json")); z = np.load(f"{D}/{name}.npz")
    eps = js["episodes"]; n = len(eps)
    steps = np.array([e["steps"] for e in eps])
    ok = [len(z["qf_off"]) == n + 1, len(z["act_off"]) == n + 1, len(z["chunk_off"]) == n + 1,
          np.array_equal(np.diff(z["qf_off"]), steps + 1), np.array_equal(np.diff(z["act_off"]), steps),
          np.array_equal(z["success"], [e["success"] for e in eps]),
          [str(x) for x in z["tasks"][z["task"]]] == [e["task"] for e in eps],
          z["qf"].shape[1] == 9, z["act"].shape[1] == 7, z["chunk"].ndim == 3,
          all((z["chunk_t"][z["chunk_off"][k]:z["chunk_off"][k + 1]] % js["n_exec"] == 0).all() for k in range(n))]
    print(name, "npz checks", ok, "perturb", js["perturb"], (eps[0].get("perturb") or {}))
EOF
echo "end=$(date -Is)"
