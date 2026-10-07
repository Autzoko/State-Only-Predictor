#!/usr/bin/env bash
# Push the working tree (code only, no data/outputs) to a server.
#   scripts/remote/sync.sh 127 | 3090 | jubail
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:?host: 127|3090|jubail}" in
  127)    DEST="NYUAIRLAB-127:~/langtian/SOPT" ;;
  3090)   DEST="NYUAIRLAB-3090:~/Desktop/langtian/SOPT" ;;
  jubail) DEST="JUBAIL:/scratch/ll5582/SOPT" ;;
  *) echo "unknown host $1"; exit 1 ;;
esac
rsync -az --delete --exclude data/ --exclude outputs/ --exclude wandb/ --exclude '__pycache__/' \
  --exclude '*.egg-info/' --exclude .pytest_cache/ ./ "$DEST/"
echo "synced to $DEST ($(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty))"
