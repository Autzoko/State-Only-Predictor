#!/bin/bash
# Data validation DV2: model-free temporal redundancy of DROID proprioception (scripts/dv_redundancy.py), CPU only.
#   sbatch scripts/slurm/dv_redundancy.sbatch
#SBATCH --job-name=sopt-dv2
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ll5582/SOPT/logs/%x-%j.out
source /etc/profile
set -uo pipefail
cd /scratch/ll5582/SOPT
export PATH=/scratch/ll5582/RoboticsNAS/env/bin:$PATH PYTHONPATH=/scratch/ll5582/SOPT/pydeps:$PWD/src
export SOPT_DATA=/scratch/ll5582/SOPT_DATA OPENBLAS_NUM_THREADS=8
echo "host=$(hostname) commit=$(git rev-parse --short HEAD) start=$(date -Is)"
python scripts/dv_redundancy.py --data-root $SOPT_DATA/processed/droid --splits train test --boot 1000 \
  --out outputs/dv2/redundancy.json; echo "dv2 exit $? $(date -Is)"
