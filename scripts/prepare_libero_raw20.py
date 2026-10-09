#!/usr/bin/env python
"""20 Hz LIBERO-90 joints / fingers / actions (no images), for inverse dynamics and closed-loop policies.

  python scripts/prepare_libero_raw20.py --sopt-data $SOPT_DATA
"""

import argparse
from pathlib import Path

from sopt.data.libero import build_raw20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sopt-data", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    d = Path(args.sopt_data) / "processed"
    eps = build_raw20(d / "libero90_raw20", d / "libero90", args.workers)
    print(eps.groupby("split").agg(episodes=("length", "size"), frames=("length", "sum")))


if __name__ == "__main__":
    main()
