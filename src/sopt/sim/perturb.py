"""E37: controlled evaluation-time perturbations of LIBERO scenes (dimensions follow LIBERO-Plus, Fei et al. 2026).

Visual perturbations change only what the cameras see; the dynamics, object layout and success check are untouched:
  camera   agentview rotated about the vertical axis through the point it looks at on the table (yaw, random sign)
  light    scene lights dimmed and tinted
  texture  table / floor / wall materials recoloured (task objects and the robot are never touched)
  noise    i.i.d. Gaussian pixel noise on both camera images
Proprioceptive control:
  robot    arm joints of the initial state offset by uniform noise (the robot starts from an unseen configuration)

Every random draw is a pure function of (kind, level, task, init) -- and of the step for pixel noise -- so all arms of
a comparison see exactly the same perturbed episode. Levels 1-3 are fixed here, before any perturbed result is seen.
"""

from __future__ import annotations

import hashlib

import numpy as np

KINDS = ("camera", "light", "texture", "noise", "robot")
LEVELS = {
    "camera": {1: 10.0, 2: 20.0, 3: 30.0},  # yaw in degrees
    "light": {1: (0.6, 0.15), 2: (0.4, 0.3), 3: (0.25, 0.45)},  # (diffuse scale, max per-channel tint)
    "texture": {1: 0.35, 2: 0.65, 3: 0.95},  # blend weight towards a random colour
    "noise": {1: 8.0, 2: 16.0, 3: 32.0},  # pixel std on the 0-255 scale
    "robot": {1: 0.05, 2: 0.10, 3: 0.20},  # max |offset| per arm joint, rad
}
# Material-name substrings of the arena surfaces recoloured by "texture" (checked per scene by scripts/e37_probe.py).
ARENA_MATERIALS = ("floor", "wall", "table", "counter")


def parse_spec(spec: str | None) -> tuple[str, int] | None:
    """'camera:2' -> ('camera', 2); None / '' / 'none' -> None."""
    if not spec or spec == "none":
        return None
    kind, level = spec.split(":")
    level = int(level)
    assert kind in KINDS and level in LEVELS[kind], spec
    return kind, level


def episode_seed(spec: tuple[str, int], task: str, init: int) -> int:
    h = hashlib.sha1(f"e37:{spec[0]}:{spec[1]}:{task}:{init}".encode()).hexdigest()
    return int(h[:15], 16)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, MuJoCo order (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_pivot(pos: np.ndarray, quat: np.ndarray, table_z: float) -> np.ndarray:
    """Where the camera's optical axis (-z of the MuJoCo camera frame) meets the plane z = table_z."""
    fwd = -_quat_to_mat(quat)[:, 2]
    assert fwd[2] < -1e-3, "camera does not look down at the table"
    return pos + fwd * (table_z - pos[2]) / fwd[2]


def yaw_about(pos: np.ndarray, quat: np.ndarray, pivot: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a world-frame camera pose about the vertical axis through pivot."""
    a = np.deg2rad(deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    qz = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    return pivot + Rz @ (pos - pivot), _quat_mul(qz, quat)


def arena_materials(model) -> list[int]:
    names = [model.material(i).name for i in range(model.nmat)]
    return [i for i, n in enumerate(names) if any(s in n.lower() for s in ARENA_MATERIALS)]


def apply_scene(env, spec: tuple[str, int] | None, seed: int, table_z: float) -> dict:
    """Perturb the MuJoCo model / initial state of a freshly reset LIBERO env (env = OffScreenRenderEnv).
    Call after set_init_state and before the settle steps. Returns the realised parameters (logged)."""
    if spec is None or spec[0] == "noise":
        return {}
    kind, level = spec
    rng = np.random.default_rng(seed)
    sim = env.sim
    m = sim.model
    p = LEVELS[kind][level]
    out = {}
    if kind == "camera":
        cid = m.camera_name2id("agentview")
        assert m.cam_bodyid[cid] == 0, "agentview must be a world-frame camera"
        pos, quat = np.array(m.cam_pos[cid]), np.array(m.cam_quat[cid])
        deg = p * rng.choice([-1.0, 1.0])
        pivot = camera_pivot(pos, quat, table_z)
        m.cam_pos[cid], m.cam_quat[cid] = yaw_about(pos, quat, pivot, deg)
        out = {"yaw_deg": float(deg), "pivot": [round(float(v), 3) for v in pivot]}
    elif kind == "light":
        scale, tint = p
        col = 1.0 - tint * rng.random(3)
        for i in range(m.nlight):
            m.light_diffuse[i] = m.light_diffuse[i] * scale * col
            m.light_specular[i] = m.light_specular[i] * scale
            m.light_ambient[i] = m.light_ambient[i] * scale * col
        out = {"scale": scale, "colour": [round(float(v), 3) for v in col]}
    elif kind == "texture":
        mats = arena_materials(m)
        assert mats, "no arena materials found"
        for i in mats:
            c = rng.random(3)
            m.mat_rgba[i, :3] = (1 - p) * m.mat_rgba[i, :3] + p * c
        out = {"materials": [m.material(i).name for i in mats]}
    elif kind == "robot":
        robot = env.env.robots[0]
        idx = list(robot._ref_joint_pos_indexes)
        assert len(idx) == 7, idx
        d = rng.uniform(-p, p, size=7)
        lo, hi = m.jnt_range[robot._ref_joint_indexes].T
        sim.data.qpos[idx] = np.clip(sim.data.qpos[idx] + d, lo + 1e-3, hi - 1e-3)
        sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        out = {"dq": [round(float(v), 4) for v in d]}
    sim.forward()
    return out


def noisy_image(img: np.ndarray, spec: tuple[str, int] | None, seed: int, step: int, cam: int) -> np.ndarray:
    if spec is None or spec[0] != "noise":
        return img
    rng = np.random.default_rng([seed, step, cam])
    noise = rng.normal(0.0, LEVELS["noise"][spec[1]], size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
