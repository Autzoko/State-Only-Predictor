"""E4b data: LIBERO demos (20 Hz) -> samples for prior-based policies, direct BC and inverse dynamics.

Timing: the prior runs on a 15 Hz grid, the simulator at 20 Hz. A sample anchored at 20 Hz frame a has
  context = features at times a/20 - (C-1..0)/15  (clamped at 0 -> "at rest before the episode"),
  future  = features at times a/20 + (1..H)/15   (clamped at the last frame),
obtained by linear interpolation of the 20 Hz features — exactly what the closed-loop controller does online.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sopt.data.libero import HF_REPO, HF_REVISION, features_20hz

SIM_FPS, PRIOR_FPS = 20.0, 15.0
IMG = 128
CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def interp_frames(F: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear interpolation of (n, D) frames at fractional indices (clamped to [0, n-1])."""
    idx = np.clip(idx, 0, len(F) - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(F) - 1)
    w = (idx - lo)[:, None].astype(F.dtype)
    return (1 - w) * F[lo] + w * F[hi]


def context_indices(anchor: float, ctx_len: int) -> np.ndarray:
    return anchor - np.arange(ctx_len - 1, -1, -1) * SIM_FPS / PRIOR_FPS


def future_indices(anchor: float, horizon: int) -> np.ndarray:
    return anchor + np.arange(1, horizon + 1) * SIM_FPS / PRIOR_FPS


class Raw20:
    """20 Hz LIBERO arrays (from prepare_libero_raw20.py) with per-episode 17-d features."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.eps = pd.read_parquet(root / "episodes.parquet")
        self.q, self.fingers = np.load(root / "q.npy"), np.load(root / "fingers.npy")
        self.actions = np.load(root / "actions.npy").astype(np.float32)
        cache = root / "features.npy"
        if cache.exists():
            self.features = np.load(cache)
        else:
            self.features = np.concatenate([
                features_20hz(self.q[s : s + n].astype(np.float64), self.fingers[s : s + n].astype(np.float64))
                for s, n in zip(self.eps["start"], self.eps["length"])
            ])
            np.save(cache, self.features)

    def episode(self, row) -> tuple[np.ndarray, np.ndarray]:
        s, n = int(row.start), int(row.length)
        return self.features[s : s + n], self.actions[s : s + n]


def select_episodes(eps: pd.DataFrame, tasks: list[str], k: int | None, salt: str = "sopt-e4b") -> pd.DataFrame:
    """k demos per task (nested in k by a salted hash), for the given task languages (in this order)."""
    from sopt.data.splits import unit_hash

    sel = eps[eps["task"].isin(tasks)].copy()
    sel["_h"] = sel["episode_index"].map(lambda e: unit_hash(str(e), salt))
    sel = sel.sort_values(["task", "_h"])
    if k is not None:
        sel = sel.groupby("task").head(k)
    sel["task_id"] = sel["task"].map({t: i for i, t in enumerate(tasks)})
    return sel.drop(columns="_h").reset_index(drop=True)


def build_frame_cache(sel: pd.DataFrame, out: str | Path, workers: int = 8) -> None:
    """Decode both camera videos of the selected episodes to (frames, 2, IMG, IMG, 3) uint8, one file per episode."""
    from concurrent.futures import ThreadPoolExecutor

    import av
    from huggingface_hub import hf_hub_download
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def work(row):
        path = out / f"ep{int(row.episode_index):06d}.npy"
        if path.exists():
            return
        cams = []
        for key in CAMERAS:
            ep = int(row.episode_index)
            f = hf_hub_download(HF_REPO, f"videos/chunk-{ep // 1000:03d}/{key}/episode_{ep:06d}.mp4",
                                repo_type="dataset", revision=HF_REVISION)
            with av.open(f) as c:
                frames = [np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((IMG, IMG),
                          Image.BILINEAR)) for fr in c.decode(video=0)]
            cams.append(np.stack(frames))
        n = min(len(cams[0]), len(cams[1]), int(row.length))
        assert abs(len(cams[0]) - int(row.length)) <= 1, f"episode {row.episode_index}: video/state mismatch"
        np.save(path, np.stack([cams[0][:n], cams[1][:n]], axis=1))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, sel.itertuples()))


class PolicyDataset(Dataset):
    def __init__(self, raw: Raw20, sel: pd.DataFrame, frames_dir: str | Path, ctx_len: int, horizon: int,
                 act_chunk: int = 16, shift: int = 8):
        self.items, self.F, self.A, self.imgs, self.task = [], [], [], [], []
        for row in sel.itertuples():
            F, A = raw.episode(row)
            im = np.load(Path(frames_dir) / f"ep{int(row.episode_index):06d}.npy", mmap_mode="r")
            n = min(len(F), len(im))
            e = len(self.F)
            self.F.append(F[:n]), self.A.append(A[:n]), self.imgs.append(im), self.task.append(int(row.task_id))
            self.items += [(e, a) for a in range(n)]
        self.ctx_len, self.horizon, self.act_chunk, self.shift = ctx_len, horizon, act_chunk, shift

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        e, a = self.items[i]
        F, A = self.F[e], self.A[e]
        ctx = interp_frames(F, context_indices(a, self.ctx_len))
        fut = interp_frames(F, future_indices(a, self.horizon))
        act = A[np.minimum(np.arange(a, a + self.act_chunk), len(A) - 1)]
        img = torch.from_numpy(np.array(self.imgs[e][a])).permute(0, 3, 1, 2)  # (2, 3, H, W) uint8
        if self.shift:  # random shift augmentation (pad + crop), per camera
            img = torch.nn.functional.pad(img.float(), (self.shift,) * 4, mode="replicate")
            out = torch.empty(2, 3, IMG, IMG)
            for c in range(2):
                dx, dy = np.random.randint(0, 2 * self.shift + 1, size=2)
                out[c] = img[c, :, dy : dy + IMG, dx : dx + IMG]
            img = out.to(torch.uint8)
        return {"ctx": torch.from_numpy(ctx.astype(np.float32)), "fut": torch.from_numpy(fut.astype(np.float32)),
                "act": torch.from_numpy(act), "img": img, "task": torch.tensor(self.task[e])}


class IDMDataset(Dataset):
    """(current state, next `k` states) -> action at the native 20 Hz."""

    def __init__(self, raw: Raw20, eps: pd.DataFrame, k: int = 4):
        self.F, self.A, self.items, self.k = [], [], [], k
        for row in eps.itertuples():
            F, A = raw.episode(row)
            e = len(self.F)
            self.F.append(F), self.A.append(A)
            self.items += [(e, t) for t in range(len(F) - 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        e, t = self.items[i]
        F = self.F[e]
        nxt = F[np.minimum(np.arange(t + 1, t + 1 + self.k), len(F) - 1)]
        return {"cur": torch.from_numpy(F[t]), "nxt": torch.from_numpy(nxt), "act": torch.from_numpy(self.A[e][t])}
