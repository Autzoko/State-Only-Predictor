"""Thin LIBERO wrapper exposing what SOPT needs: joint angles, finger positions, both cameras.

LeRobot's LiberoEnv only exposes EE pose; our prior consumes joints, so we use LIBERO's OffScreenRenderEnv.
Conventions (verified against IPEC `libero_90_no_noops_lerobot`):
  - images are rotated 180 deg relative to the raw render (OpenVLA RLDS convention), so we flip both axes;
  - dataset gripper action 1 = open / 0 = close; the simulator expects -1 open / +1 close.
Config: LIBERO_CONFIG_PATH points at <SOPT_DATA>/libero_cfg (never the shared ~/.libero); assets are fetched
from the HF hub into HF_HOME.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SETTLE_STEPS = 10  # zero actions after reset so objects settle (as in OpenVLA's LIBERO evaluation)


def setup_env_vars(sopt_data: str | os.PathLike) -> None:
    sopt_data = Path(sopt_data)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_HOME", str(sopt_data / "hf_cache"))
    cfg_dir = sopt_data / "libero_cfg"
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    if not (cfg_dir / "config.yaml").exists():
        import importlib.util

        import yaml

        root = Path(importlib.util.find_spec("libero").submodule_search_locations[0]) / "libero"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
               "init_states": str(root / "init_files"), "datasets": str(sopt_data / "libero_datasets"),
               "assets": str(root / "assets")}
        (cfg_dir / "config.yaml").write_text(yaml.dump(cfg))


def dataset_to_sim_gripper(a: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * a


def suite_tasks(suite: str = "libero_90") -> list[str]:
    from libero.libero import benchmark

    b = benchmark.get_benchmark_dict()[suite]()
    return [b.get_task(i).language for i in range(b.n_tasks)]


class LiberoTaskEnv:
    def __init__(self, suite: str, task_id: int, resolution: int = 256):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        b = benchmark.get_benchmark_dict()[suite]()
        task = b.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
        self.init_states = b.get_task_init_states(task_id)
        self.language = task.language

    @staticmethod
    def parse(obs: dict) -> dict:
        return {
            "q": np.asarray(obs["robot0_joint_pos"], dtype=np.float64),
            "fingers": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            "agent": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        }

    def reset(self, init_index: int, seed: int = 0) -> dict:
        self.env.seed(seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[init_index % len(self.init_states)])
        for _ in range(SETTLE_STEPS):
            obs, *_ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        return self.parse(obs)

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        """action: 7-d in simulator convention (gripper -1 open / +1 close). Returns (obs, success)."""
        obs, _, done, _ = self.env.step(np.asarray(action, dtype=np.float64))
        return self.parse(obs), bool(done)

    def close(self) -> None:
        self.env.close()
