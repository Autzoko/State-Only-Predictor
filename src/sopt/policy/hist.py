"""History interventions at evaluation time (reviewer item 3: does the host use the motion information in the history?).

The current frame (ctx[:, -1], which every host also receives as its current state) is never changed; only the frames
before it, i.e. what the SOPT encoder / kinematic features see beyond the current state. Every intervened history is
continuous and ends at the true current frame, so no artificial jump is introduced at the last step:
  own      unchanged
  static   every frame = the current frame: no motion. In distribution: it is exactly the padded history at the first
           step of every episode (training and evaluation pad with the first frame).
  reverse  the own history played backwards and translated to end at the current frame: same speed profile and
           path shape, motion direction (velocity sign) inverted.
  swap     the history of a training demo of the same task at the same elapsed time, translated to end at the current
           frame: a plausible motion of this task and phase, but not the robot's own.
"""

from __future__ import annotations

import numpy as np

from sopt.policy.data import context_indices, interp_frames

MODES = ("own", "static", "reverse", "swap")


def intervene(ctx: np.ndarray, mode: str, other: np.ndarray | None = None) -> np.ndarray:
    """ctx (C, D) own history (last row = current frame); other (C, D) the swap history. Returns a new (C, D)."""
    cur = ctx[-1]
    if mode == "own":
        out = ctx.copy()
    elif mode == "static":
        out = np.broadcast_to(cur, ctx.shape).copy()
    elif mode == "reverse":
        out = ctx[::-1] + (cur - ctx[0])
    elif mode == "swap":
        assert other is not None and other.shape == ctx.shape, "swap needs a same-shape history"
        out = other + (cur - other[-1])
    else:
        raise ValueError(mode)
    out[-1] = cur  # exact (the translation reproduces it only up to float rounding)
    return out.astype(ctx.dtype, copy=False)


def demo_history(feats: np.ndarray, t: int, C: int) -> np.ndarray:
    """History window of a demo (20 Hz features, (T, D)) at elapsed step t, built exactly like the rollout's own
    history (`hist` holds t + 1 frames at step t); the demo's last frame is held once it has ended."""
    return interp_frames(feats[: min(t, len(feats) - 1) + 1], context_indices(min(t, len(feats) - 1), C))
