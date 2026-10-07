"""Convert DROID (LeRobot v2.1 parquet, `cadene/droid_1.0.1`) into flat arrays.

Only state columns (plus per-episode split metadata) are kept; videos are never downloaded. From the hub,
chunks (1000 episodes, ~126 MB parquet) are streamed: download -> extract -> delete, so raw data never piles up.

Output directory:
  states.npy        float32 (N, 14)  [joint_position(7), gripper_position(1), cartesian_position(6)]
  episodes.parquet  one row per episode: episode_index, start, length, building, collector_id, success, language
  parts/            per-chunk intermediate results (resumable; removed after merging)

Note: in this port `task_category` duplicates `building` and `date` duplicates `collector_id`; we keep only
the columns that are meaningful.
"""

from __future__ import annotations

import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HF_REPO = "cadene/droid_1.0.1"
HF_REVISION = "56b622ac23335c2bb5909e1e3c72416033c65acf"  # pinned 2025-03-20
NUM_CHUNKS = 96  # 95,600 episodes, 1000 per chunk

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


def _first(col: pa.ChunkedArray):
    v = col[0].as_py()
    return v[0] if isinstance(v, list) else v


def read_episode(path: str | Path) -> tuple[np.ndarray, dict]:
    table = pq.read_table(path, columns=STATE_COLUMNS + META_COLUMNS)
    n = table.num_rows
    states = np.concatenate([_to_2d(table[c], n) for c in STATE_COLUMNS], axis=1)
    meta = {
        "episode_index": int(_first(table["episode_index"])),
        "building": str(_first(table["building"])),
        "collector_id": str(_first(table["collector_id"])),
        "success": bool(_first(table["is_episode_successful"])),
        "language": str(_first(table["language_instruction"])),
        "length": n,
    }
    return states, meta


def convert_files(files: list[Path], part: Path, workers: int) -> None:
    """Parquet episode files -> one part file (states + metadata)."""
    with Pool(workers) as pool:
        results = pool.map(read_episode, map(str, files), chunksize=16)
    tmp = part.with_suffix(".tmp.npz")
    np.savez(tmp, states=np.concatenate([r[0] for r in results]))
    pd.DataFrame([r[1] for r in results]).to_parquet(part.with_suffix(".parquet"), index=False)
    tmp.rename(part)  # the .npz marks the part as complete


def fetch_chunk(chunk: int, out: Path, workers: int, max_episodes: int | None = None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    part = out / "parts" / f"chunk_{chunk:03d}.npz"
    if part.exists():
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        if max_episodes:
            for i in range(max_episodes):
                f = f"data/chunk-{chunk:03d}/episode_{chunk * 1000 + i:06d}.parquet"
                hf_hub_download(HF_REPO, f, repo_type="dataset", revision=HF_REVISION, local_dir=tmp)
        else:
            snapshot_download(HF_REPO, repo_type="dataset", revision=HF_REVISION, local_dir=tmp,
                              allow_patterns=[f"data/chunk-{chunk:03d}/*"], max_workers=workers)
        files = sorted(Path(tmp).glob("data/chunk-*/episode_*.parquet"))
        convert_files(files, part, workers)
        shutil.rmtree(Path(tmp) / ".cache", ignore_errors=True)


def merge_parts(out: Path, keep_parts: bool = False) -> pd.DataFrame:
    parts = sorted((out / "parts").glob("chunk_*.npz"))
    states = np.concatenate([np.load(p)["states"] for p in parts])
    episodes = pd.concat([pd.read_parquet(p.with_suffix(".parquet")) for p in parts], ignore_index=True)
    episodes["start"] = np.concatenate([[0], np.cumsum(episodes["length"].to_numpy())[:-1]])
    assert episodes["length"].sum() == len(states)
    np.save(out / "states.npy", states)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    if not keep_parts:
        shutil.rmtree(out / "parts")
    return episodes


def build_from_hub(out: str | Path, chunks: list[int] | None, workers: int, max_episodes: int | None = None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = list(range(NUM_CHUNKS)) if chunks is None else chunks
    for i, c in enumerate(chunks):
        fetch_chunk(c, out, workers, max_episodes)
        print(f"chunk {c:03d} done ({i + 1}/{len(chunks)})", flush=True)
    return merge_parts(out)


def build_from_local(src: str | Path, out: str | Path, workers: int) -> pd.DataFrame:
    """For an existing local copy laid out as <src>/data/chunk-*/episode_*.parquet."""
    out = Path(out)
    for chunk_dir in sorted(Path(src).glob("data/chunk-*")):
        part = out / "parts" / f"chunk_{int(chunk_dir.name.split('-')[1]):03d}.npz"
        part.parent.mkdir(parents=True, exist_ok=True)
        if not part.exists():
            convert_files(sorted(chunk_dir.glob("episode_*.parquet")), part, workers)
    return merge_parts(out)
