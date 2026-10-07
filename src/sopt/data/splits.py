"""Leakage-safe splits: whole buildings (scenes) go to exactly one split, by a salted hash."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _unit_hash(key: str, salt: str) -> float:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def assign_splits(episodes: pd.DataFrame, val_frac: float, test_frac: float, salt: str) -> np.ndarray:
    """Returns an array of 'train' / 'val' / 'test' per episode, grouped by `building`."""
    u = episodes["building"].map(lambda b: _unit_hash(b, salt)).to_numpy()
    return np.where(u < test_frac, "test", np.where(u < test_frac + val_frac, "val", "train"))


def subsample_fraction(episodes: pd.DataFrame, fraction: float, salt: str) -> np.ndarray:
    """Nested episode subsets for data-scaling runs (1% subset is contained in the 10% subset)."""
    if fraction >= 1.0:
        return np.ones(len(episodes), dtype=bool)
    u = episodes["episode_index"].map(lambda e: _unit_hash(str(e), salt + ":frac")).to_numpy()
    return u < fraction
