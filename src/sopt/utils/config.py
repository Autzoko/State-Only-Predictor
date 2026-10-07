"""Config = base yaml merged with any number of overlay yamls and `key=value` dotlist overrides."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(overlays: list[str] | None = None, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    for path in overlays or []:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.set_readonly(cfg, True)
    return cfg
