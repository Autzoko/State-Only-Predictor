from __future__ import annotations

import numpy as np
import pandas as pd


def compute_stats(
    features: np.ndarray, episodes: pd.DataFrame, horizon: int, max_samples: int = 2_000_000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Per-dim mean/std of features and std of k-step deltas (k ~ U[1, horizon]); train episodes only."""
    rng = np.random.default_rng(seed)
    starts, lengths = episodes["start"].to_numpy(), episodes["length"].to_numpy()
    ep = rng.choice(len(episodes), size=max_samples, p=lengths / lengths.sum())
    k = rng.integers(1, horizon + 1, size=max_samples)
    ok = lengths[ep] > k
    ep, k = ep[ok], k[ok]
    t = (rng.random(len(ep)) * (lengths[ep] - k)).astype(np.int64)
    x0 = features[starts[ep] + t]
    x1 = features[starts[ep] + t + k]
    d = x1 - x0
    # Per-step scale (H, D): std of k-step deltas for each k. Used when model.per_step_norm (E4b showed the
    # pooled scale drowns near-term steps in sampling noise).
    per_k = np.stack([d[k == j].std(0) if (k == j).sum() > 1 else d.std(0) for j in range(1, horizon + 1)])
    return {
        "mean": x0.mean(0).astype(np.float32),
        "std": np.maximum(x0.std(0), 1e-3).astype(np.float32),
        "delta_std": np.maximum(d.std(0), 1e-4).astype(np.float32),
        "delta_std_k": np.maximum(per_k, 1e-5).astype(np.float32),
    }
