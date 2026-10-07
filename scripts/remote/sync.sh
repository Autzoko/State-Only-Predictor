#!/usr/bin/env bash
# Code goes through GitHub: push locally, then fast-forward the server checkout.
#   scripts/remote/sync.sh 127 | 3090 | jubail | all
set -euo pipefail
cd "$(dirname "$0")/../.."
git diff --quiet && git diff --cached --quiet || { echo "commit your changes first"; exit 1; }
git push -q origin HEAD
pull() { ssh -o BatchMode=yes "$1" "bash -lc 'cd $2 && git pull -q --ff-only && git log --oneline -1'"; }
case "${1:?127|3090|jubail|all}" in
  127)    pull NYUAIRLAB-127 '~/langtian/SOPT' ;;
  3090)   pull NYUAIRLAB-3090 '~/Desktop/langtian/SOPT' ;;
  jubail) pull JUBAIL /scratch/ll5582/SOPT ;;
  all)    pull NYUAIRLAB-127 '~/langtian/SOPT'; pull NYUAIRLAB-3090 '~/Desktop/langtian/SOPT' ;;
esac
