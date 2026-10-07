#!/usr/bin/env bash
# Create a conda env at <repo>/../SOPT_ENV (inside the user's own work dir, not the shared env list),
# from conda-forge only, and install this repo in editable mode.
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=${PREFIX:-$(cd .. && pwd)/SOPT_ENV}
CONDA=""
for c in "$(command -v conda || true)" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA=$c; break; fi
done
: "${CONDA:?conda not found}"
[ -x "$PREFIX/bin/python" ] || "$CONDA" create -y -q -p "$PREFIX" -c conda-forge --override-channels python=3.11 pip
"$PREFIX/bin/pip" install -q -e ".[dev]"
"$PREFIX/bin/python" -c "import torch, sopt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
