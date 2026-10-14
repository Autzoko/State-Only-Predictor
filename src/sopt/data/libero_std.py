"""Standard LIBERO suites (Spatial / Object / Goal / Long = libero_10) for the benchmark comparison (E13).

Source: the IPEC-COMMUNITY `*_no_noops_1.0.0_lerobot` ports (OpenVLA no-op filtering, 256 px, 20 Hz). They store
only the 8-d EE state [eef pos (world), axis-angle, 2 finger joints], no joint angles, so these experiments use an
EE-only SOPT (`model.ee_only`, joint dims hidden). The EE state is mapped into DROID's convention (flange pose in
the robot base frame) with a transform calibrated on IPEC libero_90, which has both joints and EE state:
    R_ee = R_flange @ Rz(-45 deg),   p_ee = p_flange + R_flange @ [0, 0, 0.0965] + base(task)
    position residual 0.27 mm RMS, rotation offset constant to 0.05 deg (max). The robot mount `base` differs
    between scenes ((-0.75, 0, 0.912), (-0.66, 0, 0.912), (-0.51, 0, 0.42)); configs/libero_std_base.json holds it
    for every task, measured in simulation from the robot's own joints (scripts/libero_std_base.py).
The joint slots of the 17-d features are set to 0 here and in closed-loop evaluation (`hide_joints`), so the
policy never sees joint angles in either place.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sopt.data.droid import _to_2d
from sopt.data.features import Q, raw_to_features
from sopt.data.libero import GRIPPER_OPEN_WIDTH
from sopt.utils.rotation import matrix_to_euler_xyz

SUITES = {
    "libero_spatial": ("IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot", "bf14d6258218d12c2e3c1a3b9922e163cdf6455d"),
    "libero_object": ("IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot", "15657dac2ad1c01b4e94bf54ab0493b46a8d63f9"),
    "libero_goal": ("IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot", "222cf888ed360fad0a5f983748c1cc40743d43e7"),
    "libero_10": ("IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot", "e1a223d30b896c1613f270a2bfc63d382b3de7e1"),
}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # OpenVLA protocol
# Episodes whose upstream video is corrupt (sha256 matches the hub's LFS hash, so re-downloading cannot fix it):
# dropped from the usable episode list. Training demos only; evaluation uses the simulator's init states.
CORRUPT = {"libero_goal": {82}}  # wrist_image/episode_000082.mp4: InvalidDataError on decode
TOOL = np.array([0.0, 0.0, 0.0965])
_c, _s = np.cos(np.radians(-45.0)), np.sin(np.radians(-45.0))
RZ = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])  # R_ee = R_flange @ RZ


def hide_joints(F: np.ndarray) -> np.ndarray:
    F = np.array(F, dtype=np.float32, copy=True)
    F[..., Q] = 0.0
    return F


def task_bases() -> dict[str, dict[str, list[float]]]:
    from sopt.utils.config import REPO_ROOT

    return json.loads((REPO_ROOT / "configs/libero_std_base.json").read_text())


def ee8_to_features(state: np.ndarray, base) -> np.ndarray:
    """(n, 8) LIBERO EE state -> (n, 17) SOPT features (DROID flange convention, joints = 0); base: robot mount."""
    from scipy.spatial.transform import Rotation

    R_ee = Rotation.from_rotvec(state[:, 3:6]).as_matrix()
    R_fl = R_ee @ RZ.T
    p_fl = state[:, :3] - np.asarray(base) - R_fl @ TOOL
    width = state[:, 6] - state[:, 7]
    g = np.clip((GRIPPER_OPEN_WIDTH - width) / GRIPPER_OPEN_WIDTH, 0.0, 1.0)
    raw = np.concatenate([np.zeros((len(state), 7)), g[:, None], p_fl, matrix_to_euler_xyz(R_fl)], axis=1)
    return hide_joints(raw_to_features(raw.astype(np.float32)))


def build(suite: str, out: str | Path, workers: int = 16) -> pd.DataFrame:
    """features.npy (N, 17), actions.npy (N, 7; gripper 1 = open), episodes.parquet (start / length / task)."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    repo, rev = SUITES[suite]
    out = Path(out)
    if (out / "episodes.parquet").exists():
        return usable(suite, pd.read_parquet(out / "episodes.parquet"))
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{repo}@{rev}/data/chunk-*/episode_*.parquet"))
    tasks_path = hf_hub_download(repo, "meta/tasks.jsonl", repo_type="dataset", revision=rev, local_dir=out / "meta")
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, open(tasks_path))}
    bases = task_bases()[suite]
    assert set(tasks.values()) == set(bases), "dataset task languages differ from the LIBERO benchmark"
    cols = ["observation.state", "action", "episode_index", "task_index"]

    def work(f: str):
        t = pq.read_table(f, columns=cols, filesystem=fs)
        n, task = t.num_rows, int(t["task_index"][0].as_py())
        F = ee8_to_features(_to_2d(t[cols[0]], n).astype(np.float64), bases[tasks[task]])
        return F, _to_2d(t[cols[1]], n).astype(np.float32), int(t["episode_index"][0].as_py()), task

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(work, files))
    eps = pd.DataFrame({"episode_index": [r[2] for r in res], "task_index": [r[3] for r in res],
                        "length": [len(r[0]) for r in res]})
    eps["task"] = eps["task_index"].map(tasks)
    eps["start"] = np.concatenate([[0], np.cumsum(eps["length"].to_numpy())[:-1]])
    np.save(out / "features.npy", np.concatenate([r[0] for r in res]))
    np.save(out / "actions.npy", np.concatenate([r[1] for r in res]))
    eps.to_parquet(out / "episodes.parquet", index=False)
    return usable(suite, eps)


def usable(suite: str, eps: pd.DataFrame) -> pd.DataFrame:
    return eps[~eps["episode_index"].isin(CORRUPT.get(suite, ()))].reset_index(drop=True)


class StdRaw:
    """Same interface as policy.data.Raw20 (eps + episode(row) -> (features, actions)) for a standard suite."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.eps = usable(root.name, pd.read_parquet(root / "episodes.parquet"))
        self.features = np.load(root / "features.npy")
        self.actions = np.load(root / "actions.npy")

    def episode(self, row) -> tuple[np.ndarray, np.ndarray]:
        s, n = int(row.start), int(row.length)
        return self.features[s : s + n], self.actions[s : s + n]

    def tasks(self) -> list[str]:
        """Task languages ordered by task_index (the suite's benchmark order)."""
        return list(self.eps.drop_duplicates("task_index").sort_values("task_index")["task"])
