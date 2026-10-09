"""In-memory trajectory store and fixed-length window sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sopt.data.features import GRIPPER, raw_to_features
from sopt.data.splits import unit_hash as hash_unit
from sopt.data.splits import assign_splits, subsample_fraction


class TrajectoryStore:
    """All frames of all episodes as one (N, 17) feature array; features are cached next to the raw data."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.episodes = pd.read_parquet(root / "episodes.parquet")
        cache = root / "features.npy"
        if cache.exists():
            self.features = np.load(cache, mmap_mode="r")
        else:
            self.features = raw_to_features(np.load(root / "states.npy"))
            np.save(cache, self.features)
        self._root = root
        self._next_event: np.ndarray | None = None

    @property
    def next_event(self) -> np.ndarray:
        """Global frame index of the next gripper open/close event after each frame (episode end if none).

        Gripper is binarized at 0.5; an event is a frame whose binarized state differs from the previous frame.
        Used as the "keyframe" goal (e.g. the grasp pose a VLM could point at). Cached as next_event.npy.
        """
        if self._next_event is None:
            cache = self._root / "next_event.npy"
            if cache.exists():
                self._next_event = np.load(cache, mmap_mode="r")
            else:
                closed = np.asarray(self.features[:, GRIPPER.start]) > 0.5
                out = np.empty(len(closed), dtype=np.int64)
                for s, L in zip(self.episodes["start"].to_numpy(), self.episodes["length"].to_numpy()):
                    c = closed[s : s + L]
                    events = np.flatnonzero(c[1:] != c[:-1]) + 1  # local frame indices of state changes
                    events = np.append(events, L - 1)
                    t = np.arange(L)
                    out[s : s + L] = s + events[np.searchsorted(events, t, side="right").clip(max=len(events) - 1)]
                np.save(cache, out)
                self._next_event = out
        return self._next_event

    def split_episodes(self, data_cfg, split: str) -> pd.DataFrame:
        """Datasets with a stored `split` column (e.g. LIBERO, split by task) use it; DROID uses site splits."""
        eps = self.episodes
        if "split" in eps.columns:
            keep = (eps["split"] == split).to_numpy().copy()  # pandas copy-on-write arrays are read-only
        else:
            keep = assign_splits(eps, data_cfg) == split
        if split == "train":
            keep &= subsample_fraction(eps, data_cfg.train_fraction, data_cfg.split_salt)
            k = data_cfg.get("demos_per_task")
            if k is not None:  # nested few-shot subsets: the first k episodes of each task by salted hash
                rank = (
                    eps.assign(_h=eps["episode_index"].map(lambda e: hash_unit(str(e), data_cfg.split_salt)))
                    .groupby("task_index")["_h"].rank(method="first")
                    .to_numpy()
                )
                keep &= rank <= k
        if data_cfg.success_only:
            keep &= eps["success"].to_numpy()
        return eps[keep].reset_index(drop=True)


class WindowDataset(Dataset):
    """Windows of `window` frames. Short episodes are padded with their last frame (mask=False).

    With `speed_aug=(lo, hi)` a window is resampled at a random speed factor by linear interpolation.
    With `left_pad > 0`, windows may start up to `left_pad` frames before the episode: those frames repeat the
    first frame ("robot at rest before the episode") and count as valid context. Needed for short episodes
    (LIBERO demos are ~110 frames at 15 Hz, shorter than a 128-frame window).
    """

    def __init__(
        self,
        features: np.ndarray,
        episodes: pd.DataFrame,
        window: int,
        stride: int = 1,
        speed_aug: tuple[float, float] | None = None,
        speed_aug_prob: float = 0.0,
        next_event: np.ndarray | None = None,
        left_pad: int = 0,
    ):
        self.features = features
        self.next_event = next_event
        assert 0 <= left_pad < window, "left_pad must leave at least one real frame in the window"
        self.left_pad = left_pad
        self.window = window
        self.speed_aug = speed_aug
        self.speed_aug_prob = speed_aug_prob
        ep_starts, ep_lens = episodes["start"].to_numpy(), episodes["length"].to_numpy()
        n_items = (np.maximum(ep_lens - window, 0) + left_pad) // stride + 1
        self.item_ep = np.repeat(np.arange(len(episodes)), n_items).astype(np.int32)
        offsets = np.arange(n_items.sum()) - np.repeat(np.cumsum(n_items) - n_items, n_items)
        self.item_t = (offsets * stride - left_pad).astype(np.int32)
        self.ep_starts, self.ep_lens = ep_starts, ep_lens

    def __len__(self) -> int:
        return len(self.item_ep)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.item_ep[idx]
        s0, L, t = int(self.ep_starts[e]), int(self.ep_lens[e]), int(self.item_t[idx])
        W = self.window
        speed = 1.0
        if self.speed_aug and t >= 0 and np.random.rand() < self.speed_aug_prob:
            speed = float(np.exp(np.random.uniform(np.log(self.speed_aug[0]), np.log(self.speed_aug[1]))))
            if (W - 1) * speed > L - 1:
                speed = 1.0
        if t < 0:  # left padding: repeat the first frame
            pre = -t
            real = np.asarray(self.features[s0 : s0 + min(W - pre, L)])
            seg = np.concatenate([np.repeat(real[:1], pre, axis=0), real])
            n_valid = len(seg)
            src = s0 + np.concatenate([np.zeros(pre, dtype=np.int64), np.arange(len(real))])
        elif speed == 1.0:
            seg = np.asarray(self.features[s0 + t : s0 + min(t + W, L)])
            n_valid = len(seg)
            src = s0 + t + np.arange(n_valid)
        else:
            t = min(t, int(np.floor(L - 1 - (W - 1) * speed)))
            pos = t + speed * np.arange(W)
            lo = np.minimum(np.floor(pos).astype(np.int64), L - 2)
            w = (pos - lo)[:, None].astype(np.float32)
            frames = np.asarray(self.features[s0 + lo[0] : s0 + lo[-1] + 2])
            seg = (1 - w) * frames[lo - lo[0]] + w * frames[lo - lo[0] + 1]
            n_valid = W
            src = s0 + lo
        x = np.empty((W, seg.shape[1]), dtype=np.float32)
        x[:n_valid] = seg
        x[n_valid:] = seg[-1]
        mask = np.zeros(W, dtype=bool)
        mask[:n_valid] = True
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask)}
        if self.next_event is not None:
            # Keyframe goal per frame: the state at the next gripper event (in source-frame time).
            ev = np.asarray(self.next_event[src])
            kf = np.empty_like(x)
            kf[:n_valid] = self.features[ev]
            kf[n_valid:] = kf[n_valid - 1]
            dt = np.zeros(W, dtype=np.float32)
            dt[:n_valid] = ev - src
            out["kf"], out["kf_dt"] = torch.from_numpy(kf), torch.from_numpy(dt)
        return out
