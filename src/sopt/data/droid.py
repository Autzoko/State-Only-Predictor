"""Build the DROID state dataset from the LeRobot v3.0 port (`cadene/droid_1.0.1_v30`).

Only state columns and per-episode split metadata are read: parquet column projection over HTTP range requests
fetches ~21% of each data file (~1.4 GB in total) and videos are never touched. Nothing raw is written to disk.

Output directory:
  states.npy        float32 (N, 14)  [joint_position(7), gripper_position(1), cartesian_position(6)]
  episodes.parquet  one row per episode: episode_index, start, length, building, collector_id, success, language
  parts/            per-file intermediate results (resumable; removed after merging)

Note: in this port `task_category` duplicates `building` and `date` duplicates `collector_id`; we keep only
the columns that are meaningful.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HF_REPO = "cadene/droid_1.0.1_v30"
HF_REVISION = "421fe53b89b073df8c9231d03650afd71f53c79e"  # pinned 2025-04-23; 95,584 episodes, 2048 data files

STATE_COLUMNS = [
    "observation.state.joint_position",
    "observation.state.gripper_position",
    "observation.state.cartesian_position",
]
META_COLUMNS = ["episode_index", "building", "collector_id", "is_episode_successful", "language_instruction"]


def _to_2d(col: pa.ChunkedArray, n: int) -> np.ndarray:
    col = col.combine_chunks()
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
        col = pc.list_flatten(col)
    return col.to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)


def _scalar(col: pa.ChunkedArray) -> np.ndarray:
    col = col.combine_chunks()
    if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
        col = pc.list_flatten(col)
    return col.to_numpy(zero_copy_only=False)


def read_file(path: str, filesystem=None) -> tuple[np.ndarray, pd.DataFrame]:
    """One parquet file (many episodes, rows sorted by episode) -> states + one metadata row per episode."""
    table = pq.read_table(path, columns=STATE_COLUMNS + META_COLUMNS, filesystem=filesystem, pre_buffer=True)
    n = table.num_rows
    states = np.concatenate([_to_2d(table[c], n) for c in STATE_COLUMNS], axis=1)
    ep = _scalar(table["episode_index"]).astype(np.int64)
    assert (np.diff(ep) >= 0).all(), f"rows not sorted by episode in {path}"
    first = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    meta = pd.DataFrame({
        "episode_index": ep[first],
        "building": _scalar(table["building"])[first].astype(str),
        "collector_id": _scalar(table["collector_id"])[first].astype(str),
        "success": _scalar(table["is_episode_successful"])[first].astype(bool),
        "language": _scalar(table["language_instruction"])[first].astype(str),
        "length": np.diff(np.r_[first, n]),
    })
    return states, meta


def _part_path(out: Path, rel: str) -> Path:
    chunk, file = Path(rel).parent.name, Path(rel).stem  # chunk-000 / file-000
    return out / "parts" / f"{chunk}_{file}.npz"


def build(out: str | Path, src: str | None = None, workers: int = 16, max_files: int | None = None) -> pd.DataFrame:
    """src=None streams from the HF hub; otherwise a local LeRobot v3.0 copy (<src>/data/chunk-*/file-*.parquet)."""
    out = Path(out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    if src is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        root = f"datasets/{HF_REPO}@{HF_REVISION}"
        files = sorted(fs.glob(f"{root}/data/chunk-*/file-*.parquet"))
        rels = [f.split(f"{HF_REVISION}/", 1)[1] for f in files]
    else:
        fs = None
        files = [str(f) for f in sorted(Path(src).glob("data/chunk-*/file-*.parquet"))]
        rels = [str(Path(f).relative_to(src)) for f in files]
    if not files:
        raise FileNotFoundError("no data/chunk-*/file-*.parquet found")
    files, rels = files[:max_files], rels[:max_files]

    def work(i: int) -> None:
        part = _part_path(out, rels[i])
        if part.exists():
            return
        states, meta = read_file(files[i], fs)
        meta.to_parquet(part.with_suffix(".parquet"), index=False)
        tmp = part.with_suffix(".tmp.npz")
        np.savez(tmp, states=states)
        tmp.rename(part)  # the .npz marks the part as complete

    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(work, range(len(files)))):
            if (k + 1) % 50 == 0 or k + 1 == len(files):
                print(f"  {k + 1}/{len(files)} files", flush=True)
    return merge_parts(out, [_part_path(out, r) for r in rels])


def _resolve_duplicates(episodes: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows sharing an episode_index: identical metadata -> one episode split across files (merge);
    conflicting metadata -> stray segment (keep the longest, drop the others and their frames).
    Known case @421fe53: index 88905 also labels a 59-frame BAIR fragment (info.json total_frames excludes it).
    Returns (episodes, per-frame keep mask)."""
    lengths = episodes["length"].to_numpy()
    frame_keep = np.ones(lengths.sum(), dtype=bool)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    drop = np.zeros(len(episodes), dtype=bool)
    merged_len = lengths.copy()
    key = ["building", "collector_id", "success"]
    for ep, rows in episodes.groupby("episode_index").groups.items():
        rows = np.asarray(rows)
        if len(rows) == 1:
            continue
        same_meta = (episodes.loc[rows, key].nunique() == 1).all() and (np.diff(rows) == 1).all()
        if same_meta:
            merged_len[rows[0]] = lengths[rows].sum()
            drop[rows[1:]] = True
            print(f"  episode {ep}: merged {len(rows)} segments across files", flush=True)
        else:
            keep = rows[np.argmax(lengths[rows])]
            for r in rows[rows != keep]:
                drop[r] = True
                frame_keep[starts[r] : starts[r] + lengths[r]] = False
            print(f"  WARNING episode {ep}: conflicting metadata, kept the {lengths[keep]}-frame segment, "
                  f"dropped {(lengths[rows]).sum() - lengths[keep]} frames", flush=True)
    episodes = episodes.assign(length=merged_len)[~drop].reset_index(drop=True)
    return episodes, frame_keep


def merge_parts(out: Path, parts: list[Path], keep_parts: bool = False) -> pd.DataFrame:
    states = np.concatenate([np.load(p)["states"] for p in parts])
    episodes = pd.concat([pd.read_parquet(p.with_suffix(".parquet")) for p in parts], ignore_index=True)
    assert episodes["length"].sum() == len(states)
    episodes, frame_keep = _resolve_duplicates(episodes)
    states = states[frame_keep]
    assert episodes["episode_index"].is_unique and episodes["length"].sum() == len(states)
    episodes["start"] = np.concatenate([[0], np.cumsum(episodes["length"].to_numpy())[:-1]])
    np.save(out / "states.npy", states)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    if not keep_parts:
        shutil.rmtree(out / "parts")
    return episodes
