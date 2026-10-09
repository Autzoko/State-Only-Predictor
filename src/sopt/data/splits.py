"""Leakage-safe splits at the level of *sites*.

DROID's `building` is free text (e.g. "Glen's office" / "Glens office" / "glen bureau"), so raw strings cannot be
the split unit. A site is a connected component of the bipartite graph {normalized building} <-> {collector_id}:
two buildings sharing an operator are one site. Held-out splits take whole sites, chosen by regex on their
building names (explicit lists in the config, not a hash, so the protocol is readable and fixed):

  val     OOD sites used for checkpoint selection / tuning
  test    OOD sites, touched once for final numbers
  id_val  a hash-selected fraction of episodes from training sites (in-distribution reference; shares scenes)
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd


def unit_hash(key: str, salt: str) -> float:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def _normalize_building(b: str) -> str:
    return re.sub(r"[^a-z0-9]", "", b.lower())


def site_ids(episodes: pd.DataFrame) -> np.ndarray:
    """Connected components over normalized building <-> collector_id."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nb = episodes["building"].map(_normalize_building)
    for b, c in set(zip(nb, episodes["collector_id"])):
        parent[find("B:" + b)] = find("C:" + c)
    roots = nb.map(lambda b: find("B:" + b))
    return roots.astype("category").cat.codes.to_numpy()


def assign_splits(episodes: pd.DataFrame, data_cfg) -> np.ndarray:
    """Returns 'train' / 'id_val' / 'val' / 'test' per episode."""
    sites = site_ids(episodes)
    split = np.full(len(episodes), "train", dtype=object)
    for name in ("val", "test"):
        pattern = re.compile("|".join(data_cfg[f"{name}_sites"])) if data_cfg[f"{name}_sites"] else None
        if pattern is None:
            continue
        hit = episodes["building"].map(lambda b: bool(pattern.search(b))).to_numpy()
        held = np.isin(sites, np.unique(sites[hit]))
        assert not (held & (split != "train")).any(), f"{name}_sites overlap with another held-out split"
        split[held] = name
    u = episodes["episode_index"].map(lambda e: unit_hash(str(e), data_cfg.split_salt)).to_numpy()
    split[(split == "train") & (u < data_cfg.id_val_frac)] = "id_val"
    return split.astype(str)


def subsample_fraction(episodes: pd.DataFrame, fraction: float, salt: str) -> np.ndarray:
    """Nested episode subsets for data-scaling runs (1% subset is contained in the 10% subset)."""
    if fraction >= 1.0:
        return np.ones(len(episodes), dtype=bool)
    u = episodes["episode_index"].map(lambda e: unit_hash(str(e), salt + ":frac")).to_numpy()
    return u < fraction
