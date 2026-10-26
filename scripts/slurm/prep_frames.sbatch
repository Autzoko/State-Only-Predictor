#!/bin/bash
# E39 / E40 frame cache (CPU node with network; see scripts/prep_frames.py).  sbatch scripts/slurm/prep_frames.sbatch
#SBATCH --job-name=sopt-prep
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA HF_HOME=/scratch/ll5582/SOPT_DATA/hf_cache HF_HUB_OFFLINE=0 OPENBLAS_NUM_THREADS=4
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/prep_frames.py --workers 16; echo "prep exit $? $(date -Is)"
du -sh $SOPT_DATA/processed/libero90_frames256
echo "end=$(date -Is)"
