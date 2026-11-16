# Camera-shift evaluations with the E37 / E41 file names (source this file): camera:1/2/3 = agentview yaw 10/20/30 deg
# (src/sopt/sim/perturb.py), 50 inits, written next to the run's clean rollouts_n50.json:
#   camera:1 -> e41_dose_camera1.json   camera:2 -> e37_main_camera2.json   camera:3 -> e41_dose_camera3.json
# Evaluations are independent and simulation-bound, so EVAL_PAR of them share the GPU (peak RSS ~70 GB each, as E41).
#   cam_evals <run> [gpu]        (CAM_LEVELS="1 2 3" by default; existing files are skipped)
cam_evals() {
  local r=$1 g=${2:-0} todo=() l f i
  [ -f outputs/$r/final.pt ] || { echo "cam_evals: missing outputs/$r/final.pt"; return 1; }
  for l in ${CAM_LEVELS:-1 2 3}; do
    f=$([ "$l" = 2 ] && echo e37_main_camera2 || echo e41_dose_camera$l)
    [ -f outputs/$r/$f.json ] || todo+=("$l:$f")
  done
  local P=${EVAL_PAR:-2}
  for ((i = 0; i < ${#todo[@]}; i += P)); do
    for x in "${todo[@]:i:P}"; do (l=${x%%:*}; f=${x#*:}
      CUDA_VISIBLE_DEVICES=$g python scripts/vla_eval.py --ckpt outputs/$r/final.pt --n-init 50 --perturb camera:$l \
        --out outputs/$r/$f.json > logs/$r.$f.log 2>&1
      echo "$r camera:$l exit $? $(date -Is) $(python -c "import json;print(json.load(open('outputs/$r/$f.json'))['success_rate'])" 2>/dev/null)") &
    done; wait
  done; }
