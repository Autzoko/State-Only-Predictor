#!/bin/bash
# SOPT on the lab DGX Spark cluster (GB10, aarch64; one GPU per node). Sourced by scripts/dgx/*.sbatch.
# The CIFS share holds the packed env, LIBERO data, prior checkpoints and results (no symlinks / chmod there),
# so every job runs from node-local disk: clone the repo at $COMMIT, unpack the env, copy the data, and sync
# finished runs back to $SHARE/outputs. Submit from dgx-login: sbatch --export=ALL,COMMIT=<sha> scripts/dgx/<job>.sbatch
set -uo pipefail
SHARE=/media/cvpr/langtian/SOPT
W=/var/tmp/langtian/sopt
VENV=$W/venv REPO=$W/repo-${COMMIT:?set COMMIT}
ENVPACK=$SHARE/env/venv-py311-cu130.tgz
export SOPT_DATA=$W/data HF_HOME=$W/hf PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$W"

setup_env() {  # same absolute path on every node, so the packed venv is relocatable by construction
  if [[ ! -f $VENV/.complete && -f $ENVPACK ]]; then rm -rf "$VENV" "$W/python"; tar -C "$W" -xzf "$ENVPACK"; fi
  if [[ ! -f $VENV/.complete ]]; then
    rm -rf "$VENV" "$W/uv"
    export UV_HTTP_TIMEOUT=300 UV_PYTHON_INSTALL_DIR=$W/python UV_CACHE_DIR=$W/uv-cache
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W/uv" UV_NO_MODIFY_PATH=1 sh
    "$W/uv/uv" venv --python 3.11 "$VENV"
    "$W/uv/uv" pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu130
    # as SOPT_ENV on 127 (scripts/setup_env.sh + the LIBERO extras)
    "$W/uv/uv" pip install --python "$VENV/bin/python" numpy pandas pyarrow "omegaconf>=2.3" huggingface_hub pytest \
      "hf-libero==0.1.4" "robosuite==1.4.0" "mujoco==3.8.1" av pyyaml
    touch "$VENV/.complete"
    tar -C "$W" -czf "$ENVPACK.part" venv python && mv "$ENVPACK.part" "$ENVPACK"
  fi
  export PATH=$VENV/bin:$PATH
}

setup_repo() {
  [[ -d $REPO/.git ]] || git clone -q https://github.com/Autzoko/State-Only-Predictor.git "$REPO"
  git -C "$REPO" fetch -q origin && git -C "$REPO" checkout -q "$COMMIT" || return 1
  export PYTHONPATH=$REPO/src
  mkdir -p "$W/outputs" "$SOPT_DATA"
  [[ -e $REPO/outputs ]] || ln -s "$W/outputs" "$REPO/outputs"
  rsync -r --ignore-existing "$SHARE/data/" "$SOPT_DATA/"
  rsync -r --ignore-existing "$SHARE/ckpt/" "$W/outputs/"
}

# sync_out <run>: copy a finished run (+ its logs) to the share
sync_out() { rsync -r "$W/outputs/$1" "$SHARE/outputs/" ; cp -f "$W/outputs/$1".*log "$SHARE/outputs/" 2>/dev/null; true; }
# frames decoded on this node for new episodes are reused by later jobs
sync_frames() { rsync -r --ignore-existing "$SOPT_DATA/processed/libero90_frames/" "$SHARE/data/processed/libero90_frames/"; }
