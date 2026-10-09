"""LIBERO-90 as a second (target) domain, in the same raw 14-d layout as DROID.

Source: `IPEC-COMMUNITY/libero_90_no_noops_lerobot` (73 tasks, 3,921 demos, 20 Hz), the only LeRobot port we
found that keeps joint states. Verified: FK(joint_state) + base offset (-0.75, 0, 0.912) + tool offset 9.65 cm
along flange z reproduces `ee_state` to 0.2 mm RMS, so we compute the *flange* pose in the robot base frame from
the joints with the same FK as DROID (whose cartesian_position is exactly the flange pose).

Conversions to match DROID:
  - 20 Hz -> 15 Hz by linear interpolation in time.
  - gripper: finger positions (+w/2, -w/2), w ~ 0.08 open -> DROID scale 0 open .. 1 closed: (0.08 - w) / 0.08.
Caveat: "no_noops" removed idle frames, so LIBERO motion has fewer pauses than DROID.

Splits are by task (whole tasks held out), stored in the `split` column of episodes.parquet.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sopt.data.droid import _to_2d
from sopt.utils.franka_fk import franka_fk
from sopt.utils.rotation import matrix_to_euler_xyz

HF_REPO = "IPEC-COMMUNITY/libero_90_no_noops_lerobot"
HF_REVISION = "70696aef03def70c17917f43c5e79276b7e5fbe7"  # pinned 2025-07-01
SRC_FPS, DST_FPS = 20.0, 15.0
GRIPPER_OPEN_WIDTH = 0.08
COLUMNS = ["observation.states.joint_state", "observation.states.gripper_state", "episode_index", "task_index"]


def to_droid_raw(q: np.ndarray, fingers: np.ndarray, src_fps: float = SRC_FPS) -> np.ndarray:
    """(n, 7) joints + (n, 2) finger positions at src_fps -> (m, 14) DROID raw layout at 15 Hz."""
    n = len(q)
    t_src = np.arange(n) / src_fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / DST_FPS)
    q15 = np.stack([np.interp(t_dst, t_src, q[:, j]) for j in range(7)], axis=1)
    width = fingers[:, 0] - fingers[:, 1]
    g15 = np.clip((GRIPPER_OPEN_WIDTH - np.interp(t_dst, t_src, width)) / GRIPPER_OPEN_WIDTH, 0.0, 1.0)
    M = franka_fk(q15)
    cart = np.concatenate([M[:, :3, 3], matrix_to_euler_xyz(M[:, :3, :3])], axis=1)
    return np.concatenate([q15, g15[:, None], cart], axis=1).astype(np.float32)


def _task_split(task_index: int, val_frac: float, test_frac: float, salt: str) -> str:
    u = int(hashlib.sha1(f"{salt}:task{task_index}".encode()).hexdigest()[:12], 16) / 16**12
    return "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


def build(out: str | Path, workers: int = 16, val_frac: float = 0.15, test_frac: float = 0.15,
          salt: str = "sopt-libero-v1", max_files: int | None = None) -> pd.DataFrame:
    from huggingface_hub import HfFileSystem, hf_hub_download

    out = Path(out)
    if (out / "states.npy").exists() and (out / "episodes.parquet").exists():
        print(f"{out} already built; delete states.npy to rebuild", flush=True)
        return pd.read_parquet(out / "episodes.parquet")
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    root = f"datasets/{HF_REPO}@{HF_REVISION}"
    files = sorted(fs.glob(f"{root}/data/chunk-*/episode_*.parquet"))[:max_files]
    tasks_path = hf_hub_download(HF_REPO, "meta/tasks.jsonl", repo_type="dataset", revision=HF_REVISION,
                                 local_dir=out / "meta")
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, open(tasks_path))}

    def work(f: str):
        t = pq.read_table(f, columns=COLUMNS, filesystem=fs)
        n = t.num_rows
        raw = to_droid_raw(_to_2d(t[COLUMNS[0]], n).astype(np.float64), _to_2d(t[COLUMNS[1]], n).astype(np.float64))
        return raw, int(t["episode_index"][0].as_py()), int(t["task_index"][0].as_py())

    with ThreadPoolExecutor(workers) as ex:
        results = []
        for k, r in enumerate(ex.map(work, files)):
            results.append(r)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(files)} episodes", flush=True)
    states = np.concatenate([r[0] for r in results])
    eps = pd.DataFrame({
        "episode_index": [r[1] for r in results],
        "task_index": [r[2] for r in results],
        "length": [len(r[0]) for r in results],
    })
    eps["task"] = eps["task_index"].map(tasks)
    eps["start"] = np.concatenate([[0], np.cumsum(eps["length"].to_numpy())[:-1]])
    eps["split"] = eps["task_index"].map(lambda i: _task_split(i, val_frac, test_frac, salt))
    # Columns shared with DROID so the generic store / site code still works.
    eps["building"], eps["collector_id"], eps["success"], eps["language"] = eps["task"], "libero", True, eps["task"]
    np.save(out / "states.npy", states)
    eps.to_parquet(out / "episodes.parquet", index=False)
    return eps
