#!/usr/bin/env python
"""Standard LIBERO suite: EE-only SOPT features + actions + 256 px frame cache (both cameras).

  python scripts/prepare_libero_std.py --suite libero_spatial --sopt-data $SOPT_DATA
"""

import argparse
from pathlib import Path

from sopt.data.libero_std import SUITES, build
from sopt.policy.data import build_frame_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=list(SUITES), required=True)
    p.add_argument("--sopt-data", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    root = Path(args.sopt_data) / "processed/libero_std" / args.suite
    eps = build(args.suite, root, args.workers)
    build_frame_cache(eps, root / "frames256", workers=args.workers, size=256, repo=SUITES[args.suite][0],
                      revision=SUITES[args.suite][1])
    print(args.suite, len(eps), "episodes,", int(eps["length"].sum()), "frames,", eps["task"].nunique(), "tasks")


if __name__ == "__main__":
    main()
