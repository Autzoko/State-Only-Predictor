import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from sopt.data.dataset import WindowDataset
from sopt.data.features import raw_to_features
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


def test_energy_score():
    from sopt.eval.forecast import energy_score_pos

    fut = torch.zeros(2, 8, 17)
    det = torch.zeros(2, 1, 8, 17)
    det[..., 8] = 0.01  # 1 cm off in x at every step
    assert torch.allclose(energy_score_pos(det, fut), torch.tensor([1.0, 1.0]))
    # Two samples straddling the truth (+-1 cm): ES = 1 - 0.5 * 2 = 0, better than the 1 cm deterministic guess.
    two = torch.zeros(2, 2, 8, 17)
    two[:, 0, :, 8], two[:, 1, :, 8] = 0.01, -0.01
    assert torch.allclose(energy_score_pos(two, fut), torch.zeros(2), atol=1e-5)


def test_next_event_and_keyframe_goals(tmp_path):
    from sopt.data.dataset import TrajectoryStore

    # Two episodes; gripper (feature 7) closes at local frame 3 and opens at 6 in episode 0, never changes in 1.
    feats = np.zeros((14, 17), dtype=np.float32)
    feats[3:6, 7] = 1.0
    feats[:, 0] = np.arange(14)
    np.save(tmp_path / "features.npy", feats)
    pd.DataFrame({"start": [0, 8], "length": [8, 6], "episode_index": [0, 1], "building": ["a", "b"],
                  "collector_id": ["u", "v"], "success": [True, True]}).to_parquet(tmp_path / "episodes.parquet")
    store = TrajectoryStore(tmp_path)
    assert store.next_event.tolist() == [3, 3, 3, 6, 6, 6, 7, 7, 13, 13, 13, 13, 13, 13]
    ds = WindowDataset(store.features, store.episodes, window=4, next_event=store.next_event)
    item = ds[0]  # frames 0..3 of episode 0
    assert item["kf"][:, 0].tolist() == [3, 3, 3, 6] and item["kf_dt"].tolist() == [3, 2, 1, 3]


@pytest.mark.parametrize("objective", ["ar_flow", "masked"])
def test_goal_conditioning_starts_at_pretrained_model(objective):
    base = ["model.d_model=32", "model.n_layers=2", "model.n_heads=2", "model.head_hidden=64",
            "model.ctx_len=32", "model.horizon=8", f"model.objective={objective}"]
    feats, eps = _toy_data()
    stats = compute_stats(feats, eps, 8, max_samples=5000)
    pre = StatePrior(load_config(overrides=base).model, stats).eval()
    with torch.no_grad():  # stand-in for pretraining: an untrained flow head's AdaLN gates are exactly zero
        for prm in pre.head.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    gc = StatePrior(load_config(overrides=[*base, "model.goal_cond=true"]).model, stats).eval()
    missing, unexpected = gc.load_state_dict(pre.state_dict(), strict=False)
    assert all(k.startswith("goal_proj") or k == "goal_probs" for k in missing) and not unexpected
    ds = WindowDataset(feats, eps, pre.window)
    batch = torch.utils.data.default_collate([ds[i] for i in range(4)])
    ctx, goal = batch["x"][:, :32], batch["x"][:, -1]
    torch.manual_seed(0)
    a = pre.forecast(ctx, 2)
    torch.manual_seed(0)
    b = gc.forecast(ctx, 2, goal=goal, goal_type="endpoint")
    assert torch.allclose(a, b, atol=1e-5)
    gc.train()
    loss = gc.loss(batch["x"], batch["mask"], kf=batch["x"])["loss"]
    loss.backward()
    assert torch.isfinite(loss) and gc.goal_proj[-1].weight.grad.abs().sum() > 0


def test_left_pad_windows():
    feats, eps = _toy_data()
    eps = eps.iloc[:1].assign(length=20)
    ds = WindowDataset(feats, eps, window=16, stride=4, left_pad=8)
    first = ds[0]  # starts 8 frames before the episode
    assert first["mask"].all() and torch.equal(first["x"][0], first["x"][8])
    assert torch.equal(first["x"][8], torch.from_numpy(feats[0]))
    assert len(ds) == (20 - 16 + 8) // 4 + 1


def test_libero_conversion_matches_fk():
    from sopt.data.features import POS, ROT
    from sopt.data.libero import to_droid_raw
    from sopt.utils.franka_fk import franka_fk

    rng = np.random.default_rng(0)
    q = np.cumsum(rng.normal(0, 0.01, (41, 7)), 0) + np.array([0, -0.5, 0, -2.2, 0, 1.8, 0.8])
    fingers = np.stack([np.full(41, 0.04), np.full(41, -0.04)], 1)
    raw = to_droid_raw(q, fingers)
    assert len(raw) == 31  # 2 s at 20 Hz -> 15 Hz
    feats = raw_to_features(raw)
    M = franka_fk(raw[:, :7].astype(np.float64))
    assert np.allclose(feats[:, POS], M[:, :3, 3], atol=1e-5)
    assert np.allclose(feats[:, ROT], np.concatenate([M[:, :3, 0], M[:, :3, 1]], 1), atol=1e-5)
    assert np.allclose(raw[:, 7], 0.0)  # fully open


def test_per_step_norm_scales_near_term_smaller():
    feats, eps = _toy_data()
    stats = compute_stats(feats, eps, 8, max_samples=20000)
    assert stats["delta_std_k"].shape == (8, 17)
    assert (stats["delta_std_k"][0] < stats["delta_std_k"][-1]).mean() > 0.9  # random walk: grows with k
    cfg = load_config(overrides=["model.d_model=32", "model.n_layers=1", "model.n_heads=2", "model.ctx_len=32",
                                 "model.horizon=8", "model.per_step_norm=true"])
    m = StatePrior(cfg.model, stats)
    ds = WindowDataset(feats, eps, m.window)
    b = torch.utils.data.default_collate([ds[i] for i in range(4)])
    assert torch.isfinite(m.loss(b["x"], b["mask"])["loss"])
    assert m.forecast(b["x"][:, :32], 2).shape == (4, 2, 8, 17)
