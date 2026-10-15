#!/bin/bash
# On dgx-login: bash scripts/dgx/submit.sh <job.sbatch> [extra sbatch args]
# Pulls this clone, publishes common.sh to the share (compute nodes cannot see the login home) and submits at HEAD.
set -euo pipefail
cd "$(dirname "$0")/../.."
git pull -q --ff-only
mkdir -p /media/cvpr/langtian/SOPT/scripts_dgx
cp -f scripts/dgx/common.sh /media/cvpr/langtian/SOPT/scripts_dgx/common.sh
job=$1; shift
sbatch --export=ALL,COMMIT=$(git rev-parse HEAD) "$@" "$job"
