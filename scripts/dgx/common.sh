#!/bin/bash
# SOPT on the lab DGX Spark cluster (GB10, aarch64; one GPU per node). Sourced by scripts/dgx/*.sbatch.
# The CIFS share holds the packed env, LIBERO data, prior checkpoints and results (no symlinks / chmod there),
# so every job runs from node-local disk: clone the repo at $COMMIT, unpack the env, copy the data, and sync
# finished runs back to $SHARE/outputs. Submit from dgx-login: sbatch --export=ALL,COMMIT=<sha> scripts/dgx/<job>.sbatch
set -uo pipefail
SHARE=/media/cvpr/langtian/SOPT
W=/var/tmp/langtian/sopt
REPO=$W/repo-${COMMIT:?set COMMIT}
# ENV_KIND=base: torch + LIBERO (TrunkBC etc.); ENV_KIND=vla: + LeRobot (pinned as SOPT_VLA_ENV on 127) for SmolVLA / DP
ENV_KIND=${ENV_KIND:-base}
# (vla: torch / torchvision / transformers pinned to SOPT_VLA_ENV on 127; LeRobot 0.6.2 needs torchvision < 0.27)
if [[ $ENV_KIND = vla ]]; then VENV=$W/venv-vla PY=3.12 ENVPACK=$SHARE/env/venv-vla-py312-cu130.tgz TORCH="torch==2.11.0 torchvision==0.26.0"
else VENV=$W/venv PY=3.11 ENVPACK=$SHARE/env/venv-py311-cu130.tgz TORCH="torch torchvision"; fi
LEROBOT="lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"
export SOPT_DATA=$W/data HF_HOME=$W/hf PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# shared lab account: keep every ~/.cache, ~/.libero etc. (e.g. hf_libero assets) under our node-local dir
export HOME=$W/home
mkdir -p "$W" "$HOME"

setup_env() {  # same absolute path on every node, so the packed venv is relocatable by construction
  if [[ ! -f $VENV/.complete && -f $ENVPACK ]]; then rm -rf "$VENV"; tar -C "$W" -xzf "$ENVPACK"; fi
  if [[ ! -f $VENV/.complete ]]; then
    rm -rf "$VENV"
    export UV_HTTP_TIMEOUT=300 UV_PYTHON_INSTALL_DIR=$W/python UV_CACHE_DIR=$W/uv-cache
    [[ -x $W/uv/uv ]] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$W/uv" UV_NO_MODIFY_PATH=1 sh
    local uv="$W/uv/uv" py="$VENV/bin/python"
    "$uv" venv --python $PY "$VENV"
    "$uv" pip install --python "$py" $TORCH --index-url https://download.pytorch.org/whl/cu130
    "$uv" pip freeze --python "$py" | grep -E "^torch(vision)?==" > "$W/torch-pin.txt"  # keep the CUDA aarch64 build
    # as SOPT_ENV / SOPT_VLA_ENV on 127 (scripts/setup_env.sh + the LIBERO extras [+ LeRobot, diffusers for DP])
    "$uv" pip install --python "$py" -c "$W/torch-pin.txt" numpy pandas pyarrow "omegaconf>=2.3" huggingface_hub pytest \
      "hf-libero==0.1.4" "robosuite==1.4.0" "mujoco==3.8.1" av pyyaml
    if [[ $ENV_KIND = vla ]]; then
      "$uv" pip install --python "$py" -c "$W/torch-pin.txt" "$LEROBOT" "diffusers==0.41.0" "transformers==5.5.4"
      "$py" -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; import diffusers" || return 1
    fi
    touch "$VENV/.complete"
    tar -C "$W" -czf "$ENVPACK.part" "$(basename "$VENV")" python && mv "$ENVPACK.part" "$ENVPACK"
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
