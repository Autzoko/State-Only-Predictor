#!/usr/bin/env python
"""Robot base offset of every standard-suite LIBERO task (robot mounting differs between scenes).

For each task: reset to init state 0 and compute base = eef_pos - (p_flange + R_flange @ tool) from the simulator's
own joint angles (Franka FK in the base frame). Pure robot geometry; no demonstration data involved.
  python scripts/libero_std_base.py --out configs/libero_std_base.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sopt.data.libero_std import SUITES, TOOL
from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks
from sopt.utils.franka_fk import franka_fk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_env_vars(args.sopt_data)
    table = {}
    for suite in SUITES:
        for i, lang in enumerate(suite_tasks(suite)):
            env = LiberoTaskEnv(suite, i)
            o = env.reset(0)
            M = franka_fk(o["q"][None])[0]
            base = o["eef"] - (M[:3, 3] + M[:3, :3] @ TOOL)
            env.close()
            table.setdefault(suite, {})[lang] = [round(float(v), 4) for v in base]
            print(suite, i, lang, table[suite][lang], flush=True)
    Path(args.out).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
