import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from sopt.data.dataset import WindowDataset
from sopt.data.normalize import compute_stats
from sopt.models.state_prior import StatePrior
from sopt.utils.config import load_config


def _toy_data(n_eps=6, D=17):
    rng = np.random.default_rng(0)
    lens = rng.integers(20, 80, n_eps)
    feats = np.cumsum(rng.normal(0, 0.01, (lens.sum(), D)), 0).astype(np.float32)
    eps = pd.DataFrame({"start": np.concatenate([[0], np.cumsum(lens)[:-1]]), "length": lens})
    return feats, eps


def test_window_dataset_padding_and_speed():
    feats, eps = _toy_data()
    ds = WindowDataset(feats, eps, window=40, stride=3, speed_aug=(0.8, 1.25), speed_aug_prob=1.0)
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (40, 17) and item["mask"][0]
    short = WindowDataset(feats, eps.assign(length=10), window=40)
    item = short[0]
    assert item["mask"].sum() == 10 and torch.equal(item["x"][-1], item["x"][9])


@pytest.mark.parametrize("objective", ["ar_regression", "ar_flow", "masked"])
def test_loss_and_forecast_shapes(objective):
    cfg = load_config(overrides=[f"model.objective={objective}", "model.d_model=32", "model.n_layers=2",
                                 "model.n_heads=2", "model.head_hidden=64", "model.ctx_len=32", "model.horizon=8"])
    feats, eps = _toy_data()
    model = StatePrior(cfg.model, compute_stats(feats, eps, cfg.model.horizon, max_samples=5000))
    ds = WindowDataset(feats, eps, model.window)
    batch = torch.utils.data.default_collate([ds[i] for i in range(8)])
    loss = model.loss(batch["x"], batch["mask"], input_noise=0.01)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    pred = model.forecast(batch["x"][:, : cfg.model.ctx_len], num_samples=3)
    assert pred.shape == (8, 3, cfg.model.horizon, 17) and torch.isfinite(pred).all()


def test_resolve_duplicate_episodes():
    from sopt.data.droid import _resolve_duplicates

    eps = pd.DataFrame({
        "episode_index": [0, 1, 1, 2, 2],
        "building": ["a", "b", "b", "X", "c"],
        "collector_id": ["u"] * 5,
        "success": [True] * 5,
        "length": [3, 2, 4, 1, 5],
    })
    out, keep = _resolve_duplicates(eps)
    assert out["episode_index"].tolist() == [0, 1, 2]
    assert out["length"].tolist() == [3, 6, 5]  # 1: merged across files; 2: 1-frame stray segment dropped
    assert keep.sum() == 14 and not keep[9]


def test_site_splits_hold_out_whole_sites():
    from sopt.data.splits import assign_splits

    eps = pd.DataFrame({
        "episode_index": range(8),
        "building": ["Glen's office", "glen bureau", "Gates", "Gates", "AHG kitchen", "AHG_lab", "X", "Y"],
        "collector_id": ["g", "g", "s", "s", "a", "b", "b", "z"],
    })
    cfg = OmegaConf.create({"val_sites": ["^Glen"], "test_sites": ["^AHG kitchen"], "id_val_frac": 0.0,
                            "split_salt": "t"})
    split = assign_splits(eps, cfg)
    # "glen bureau" shares collector g with "Glen's office" -> same site; "AHG_lab" (collector b) is not linked
    # to "AHG kitchen" (collector a), so the pattern only holds out the latter.
    assert split.tolist() == ["val", "val", "train", "train", "test", "train", "train", "train"]
