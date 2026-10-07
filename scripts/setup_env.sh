#!/usr/bin/env bash
# Create the conda env `sopt` and install this repo in editable mode. Works without conda on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-sopt}
for c in "$(command -v conda || true)" "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  [ -n "$c" ] && [ -x "$c" ] && CONDA=$c && break
done
: "${CONDA:?conda not found}"
"$CONDA" env list | grep -qE "^$ENV\s" || "$CONDA" create -y -q -n "$ENV" python=3.11
"$CONDA" run -n "$ENV" pip install -q -e ".[dev]"
"$CONDA" run -n "$ENV" python -c "import torch, sopt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
