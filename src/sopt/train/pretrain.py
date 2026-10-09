"""Single-GPU pretraining loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sopt.data.augment import augment_batch
from sopt.data.dataset import TrajectoryStore, WindowDataset
from sopt.data.normalize import compute_stats
from sopt.eval.forecast import evaluate_forecast
from sopt.models.state_prior import StatePrior
from sopt.utils.misc import JsonlLogger, git_rev, seed_everything


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def build_model(model_cfg, stats) -> StatePrior:
    return StatePrior(model_cfg, stats)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[StatePrior, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg.model, ckpt["stats"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def goal_modes(model_cfg) -> tuple[str, ...]:
    return ("none", "endpoint", "keyframe") if model_cfg.get("goal_cond", False) else ("none",)


def make_loaders(cfg, store: TrajectoryStore, window: int):
    train_eps = store.split_episodes(cfg.data, "train")
    val_eps = store.split_episodes(cfg.data, "val")
    ne = store.next_event if cfg.model.get("goal_cond", False) else None
    pad = min(cfg.data.get("left_pad", 0), cfg.model.ctx_len - cfg.model.patch)  # keep >= 1 real context patch
    train_ds = WindowDataset(
        store.features, train_eps, window, cfg.data.stride, tuple(cfg.data.speed_aug), cfg.data.speed_aug_prob,
        next_event=ne, left_pad=pad,
    )
    val_ds = WindowDataset(store.features, val_eps, window, stride=cfg.data.get("eval_stride") or cfg.model.ctx_len,
                           next_event=ne, left_pad=pad)
    nw = cfg.train.num_workers
    common = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, cfg.train.batch_size, shuffle=True, drop_last=True, generator=g, **common)
    # Shuffled so `eval_batches` is a random subset, re-seeded before every evaluation (see `reset_val_order`)
    # so all evaluations of a run score the same windows.
    val_dl = DataLoader(val_ds, cfg.eval.batch_size, shuffle=True, generator=torch.Generator().manual_seed(0), **common)
    return train_eps, val_eps, train_dl, val_dl


def run(cfg) -> Path:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_root) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    store = TrajectoryStore(cfg.data.root)
    window = cfg.model.ctx_len + cfg.model.horizon
    train_eps, val_eps, train_dl, val_dl = make_loaders(cfg, store, window)
    init = cfg.get("init_from")
    if init:  # post-training: keep the pretrained normalization so the backbone sees the same inputs
        init_ckpt = torch.load(init, map_location="cpu", weights_only=False)
        stats = init_ckpt["stats"]
    else:
        stats = compute_stats(np.asarray(store.features), train_eps, cfg.model.horizon, seed=cfg.seed)
    model = build_model(cfg.model, stats).to(device)
    if init:
        missing, unexpected = model.load_state_dict(init_ckpt["model"], strict=False)
        new = [k for k in missing if not k.startswith("goal_proj") and k != "goal_probs"]
        assert not new and not unexpected, f"init_from mismatch: missing {new}, unexpected {unexpected}"
        print(f"initialized from {init} (step {init_ckpt['step']}); new params: {missing}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "git": git_rev(), "n_params": n_params, "train_episodes": len(train_eps), "val_episodes": len(val_eps),
        "train_windows": len(train_dl.dataset), "val_windows": len(val_dl.dataset), "device": str(device),
        "init_from": init,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr, betas=(0.9, 0.95),
    )
    step = 0
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"resumed from step {step}", flush=True)

    fwd = torch.compile(model.loss) if cfg.train.compile else model.loss
    amp = device.type == "cuda" and cfg.train.precision == "bf16"
    logger = JsonlLogger(out / "metrics.jsonl", cfg.train.wandb, dict(project=cfg.train.wandb_project, name=cfg.run_name,
                                                                       config=OmegaConf.to_container(cfg)))

    def save(name: str):
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step, "stats": stats,
                    "config": OmegaConf.to_container(cfg), "meta": meta}, out / name)

    best = float("inf")

    def evaluate_and_select():
        nonlocal best
        val_dl.generator.manual_seed(0)  # same val subset at every evaluation
        res = evaluate_forecast(model, val_dl, device, cfg.eval.num_samples, cfg.train.eval_batches,
                                goal_modes(cfg.model))
        logger.log(step, **{f"val/{k}": v for k, v in res.items() if not isinstance(v, list)})
        if cfg.model.get("goal_cond", False):  # select on the realistic goal type
            key = res["model@keyframe/energy_pos_cm"]
        elif cfg.train.get("select_on", "auto") == "energy":
            key = res["model/energy_pos_cm"]
        else:
            key = res.get(f"model/min{cfg.eval.num_samples}_pos_ade_cm", res["model/pos_ade_cm"])
        print(f"step {step} val pos_ade_cm model {res['model/pos_ade_cm']:.3f} "
              f"const_vel {res['const_vel/pos_ade_cm']:.3f} selection {key:.3f}", flush=True)
        if key < best:
            best = key
            save("best.pt")
        save("last.pt")

    # For post-training, score the initial model too, so early stopping can never end below "no post-training".
    if cfg.train.get("eval_at_start", False) and step == 0:
        evaluate_and_select()
    t0 = time.time()
    model.train()
    while step < cfg.train.max_steps:
        for batch in train_dl:
            x = batch["x"].to(device, non_blocking=True)
            kf = batch["kf"].to(device, non_blocking=True) if "kf" in batch else None
            if kf is None:
                x = augment_batch(x, cfg.aug.yaw_deg)
            else:
                x, kf = augment_batch(x, cfg.aug.yaw_deg, kf)
            mask = batch["mask"].to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg.train)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = fwd(x, mask, cfg.aug.input_noise, kf)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            step += 1

            if step % cfg.train.log_every == 0:
                dt = time.time() - t0
                logger.log(step, loss=loss.item(), grad_norm=gnorm.item(), lr=opt.param_groups[0]["lr"],
                           steps_per_s=cfg.train.log_every / dt)
                print(f"step {step} loss {loss.item():.4f} ({cfg.train.log_every / dt:.1f} it/s)", flush=True)
                t0 = time.time()
            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
                evaluate_and_select()
            elif step % cfg.train.save_every == 0:
                save("last.pt")
            if step >= cfg.train.max_steps:
                break
    return out
