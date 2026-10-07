#!/usr/bin/env python
"""Download the state/action parquet files of DROID 1.0.1 (LeRobot port), without videos (~12 GB vs ~400 GB).

  python scripts/download_droid.py --dest data/raw/droid_1.0.1
  python scripts/download_droid.py --dest data/raw/droid_debug --chunks 0 10 50 --max-per-chunk 10
"""

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "cadene/droid_1.0.1"
REVISION = "56b622ac23335c2bb5909e1e3c72416033c65acf"  # pinned 2025-03-20
META = ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]  # skip episodes_stats.jsonl (1 GB)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", required=True)
    p.add_argument("--chunks", type=int, nargs="*", help="only these chunk ids (1000 episodes each)")
    p.add_argument("--max-per-chunk", type=int, default=None, help="debug: first N episodes of each chunk")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if args.max_per_chunk:
        files = META + [
            f"data/chunk-{c:03d}/episode_{c * 1000 + i:06d}.parquet" for c in args.chunks for i in range(args.max_per_chunk)
        ]
        for f in files:
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=args.dest)
    else:
        patterns = META + ([f"data/chunk-{c:03d}/*" for c in args.chunks] if args.chunks else ["data/*"])
        snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=args.dest,
                          allow_patterns=patterns, max_workers=args.workers)
    print("done:", args.dest)


if __name__ == "__main__":
    main()
