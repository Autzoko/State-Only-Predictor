#!/bin/bash
# Build the project's own Python environment at /scratch/ll5582/SOPT/env (2026-11-16). Until then the jobs borrowed the
# interpreter of /scratch/ll5582/RoboticsNAS/env, which no longer exists. Versions are pinned to what that environment
# held (docs/COMPUTE_ENV.md), LeRobot from its pinned git commit; pydeps (diffusers 0.39, dm-tree, sentencepiece, ftfy)
# stays first on PYTHONPATH as before. Needs internet on the compute node. Ends with an import
# test of every third-party module the code uses and a CUDA check.
#   sbatch scripts/slurm/build_env.sbatch
#SBATCH --job-name=sopt-env
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn014,cn015,cn270,cn256
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -euo pipefail
cd /scratch/ll5582/SOPT
E=/scratch/ll5582/SOPT/env
export UV_CACHE_DIR=/scratch/ll5582/cache/uv UV_PYTHON_INSTALL_DIR=/scratch/ll5582/cache/uv_python UV_LINK_MODE=copy
UV=/home/ll5582/.local/bin/uv
LEROBOT=3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e  # the pinned LeRobot commit (version 0.6.2, 2026-09-03; not on PyPI)
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
[ -x $E/bin/python ] || $UV venv --python 3.12 $E
$UV pip install --python $E/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  "lerobot[smolvla,groot,xvla,fastwam,pi,diffusion,libero] @ git+https://github.com/huggingface/lerobot@$LEROBOT" \
  transformers==5.5.4 accelerate==1.15.0 diffusers==0.39.0 \
  hf-libero==0.1.4 robosuite==1.4.0 mujoco==3.8.1 PyOpenGL==3.1.10 egl-probe==1.0.2 \
  numpy==2.2.6 pandas==2.3.3 pyarrow==25.0.1 scipy==1.18.1 omegaconf==2.3.1 einops==0.8.2 av==15.1.0 \
  safetensors==0.8.0 huggingface_hub==1.33.0 pyyaml pillow
$UV pip freeze --python $E/bin/python > logs/sopt-env-freeze.txt
echo "installed $(wc -l < logs/sopt-env-freeze.txt) packages $(date -Is)"
export PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/scratch/ll5582/SOPT_DATA/libero_cfg
$E/bin/python - <<'PY'
import importlib
mods = ["torch", "torchvision", "numpy", "pandas", "pyarrow", "scipy", "omegaconf", "einops", "av", "PIL", "yaml",
        "transformers", "diffusers", "huggingface_hub", "mujoco", "robosuite", "libero", "tree", "sentencepiece", "ftfy",
        "lerobot.policies.diffusion.modeling_diffusion", "lerobot.policies.diffusion.processor_diffusion",
        "lerobot.policies.smolvla.modeling_smolvla", "lerobot.policies.smolvla.processor_smolvla",
        "lerobot.policies.groot.modeling_groot", "lerobot.policies.groot.processor_groot",
        "lerobot.policies.xvla.modeling_xvla", "lerobot.policies.xvla.processor_xvla",
        "lerobot.policies.fastwam.modeling_fastwam", "lerobot.policies.pi05.modeling_pi05",
        "sopt.vla.diffusion", "sopt.vla.smolvla", "sopt.vla.groot", "sopt.models.kinematic"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("ok", m, getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:  # noqa: BLE001
        bad.append(m); print("FAIL", m, type(e).__name__, e)
import torch
x = torch.randn(256, 256, device="cuda"); print("cuda", torch.cuda.get_device_name(0), float((x @ x).sum()) != 0)
print("ENV", "PASS" if not bad else f"FAIL {bad}")
PY
echo "end=$(date -Is)"
