#!/usr/bin/env bash
# Create the conda env `sopt` and install the package in editable mode.
#   bash scripts/setup_env.sh            # CUDA build of torch picked by pip
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-sopt}
if ! conda env list | grep -qE "^$ENV\s"; then
  conda create -y -n "$ENV" python=3.11
fi
conda run -n "$ENV" pip install -e ".[dev]"
conda run -n "$ENV" python -c "import torch, sopt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
