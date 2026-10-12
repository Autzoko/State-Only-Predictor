#!/usr/bin/env python
"""Failure taxonomy of traced LIBERO rollouts (scripts/e4b_eval.py --trace): where does a policy fail?

For each failed pick-and-place episode, from simulator state:
  wrong_target  the target object was never approached, but another object was moved (semantic / "where")
  no_reach      the target was never approached and nothing else moved (stalled / wandering)
  grasp_miss    the gripper got close to the target but never lifted it ("how": precision)
  place_fail    the target was lifted but the task did not complete (transport / release precision)
The drawer task has no tracked object and is reported separately.

  python scripts/failure_taxonomy.py outputs/e4b_bc_k5/trace_n50.json [more traces ...]
"""

import json
import sys
from collections import Counter

import numpy as np

TARGET = {
    "pick up the cream cheese and put it in the tray": "cream_cheese_1",
    "pick up the tomato sauce and put it in the tray": "tomato_sauce_1",
    "put the ketchup in the top drawer of the cabinet": "ketchup_1",
    "put the white bowl on top of the cabinet": "white_bowl_1",
    "put the wine bottle on the wine rack": "wine_bottle_1",
    "put the yellow and white mug on the right plate": "white_yellow_mug_1",
}
NEAR, LIFT, MOVED = 0.06, 0.03, 0.03  # m


def classify(ep: dict) -> str:
    if ep["success"]:
        return "success"
    tgt = TARGET.get(ep["task"])
    if tgt is None:
        return "drawer_fail"
    tr = ep["trace"]
    pos = np.array([r["objs"][tgt] for r in tr])
    near = np.linalg.norm(np.array([r["objs"][tgt + "_to_robot0_eef"] for r in tr]), axis=1).min() < NEAR
    lifted = (pos[:, 2] - pos[0, 2]).max() > LIFT
    if lifted:
        return "place_fail"
    if near:
        return "grasp_miss"
    others = [k for k in tr[0]["objs"] if not k.endswith("_to_robot0_eef") and k != tgt]
    moved = any(np.linalg.norm(np.array([r["objs"][k] for r in tr]) - tr[0]["objs"][k], axis=1).max() > MOVED
                for k in others)
    return "wrong_target" if moved else "no_reach"


def main():
    cats = ["success", "wrong_target", "no_reach", "grasp_miss", "place_fail", "drawer_fail"]
    print(f"{'policy':34s}" + "".join(f"{c:>13s}" for c in cats))
    for path in sys.argv[1:]:
        eps = json.load(open(path))["episodes"]
        c = Counter(classify(e) for e in eps)
        print(f"{path.split('/')[-2]:34s}" + "".join(f"{c[k]:13d}" for k in cats))
        per = {}
        for e in eps:
            per.setdefault(e["task"][:28], Counter())[classify(e)] += 1
        for t, cc in per.items():
            print(f"   {t:31s}" + "".join(f"{cc[k]:13d}" for k in cats))


if __name__ == "__main__":
    main()
