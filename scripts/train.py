#!/usr/bin/env python
"""python scripts/train.py --config configs/model/s.yaml --config configs/experiment/e1_ar_flow.yaml run_name=e1_ar_flow_s"""

import argparse

from omegaconf import OmegaConf

from sopt.train.pretrain import run
from sopt.utils.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", default=[], help="overlay yaml (repeatable)")
    p.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. train.lr=1e-4")
    args = p.parse_intermixed_args()

    cfg = load_config(args.config, args.overrides)
    print(OmegaConf.to_yaml(cfg))
    print("output:", run(cfg))


if __name__ == "__main__":
    main()
