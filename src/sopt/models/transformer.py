"""Pre-LN Transformer with rotary position embeddings over patch tokens.

Options (E17 architecture ablation; defaults = the original block, so old checkpoints load unchanged):
  mlp  'gelu' | 'swiglu' (hidden scaled by 2/3 to keep the parameter count);
  norm 'layernorm' | 'rmsnorm';
  init 'default' (PyTorch) | 'xavier' (xavier-normal weights, zero biases, residual output projections
       scaled by 1/sqrt(2 * n_layers)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rope_cache(n: int, head_dim: int, device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    ang = torch.outer(torch.arange(n, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    """qk_norm: RMSNorm on per-head q/k (Dehghani et al. 2023). Without it, the first layer's qkv weights grew
    without bound (norm ~590 vs ~18 elsewhere) and gradient norms exploded after ~25k steps (E1, 2026-10-07)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, qk_norm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        head_dim = d_model // n_heads
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, cos, sin, attn_mask):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, N, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden)
        self.out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.out(F.silu(a) * b))


NORMS = {"layernorm": nn.LayerNorm, "rmsnorm": nn.RMSNorm}


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float, qk_norm: bool = True,
                 mlp: str = "gelu", norm: str = "layernorm"):
        super().__init__()
        self.ln1 = NORMS[norm](d_model)
        self.attn = Attention(d_model, n_heads, dropout, qk_norm)
        self.ln2 = NORMS[norm](d_model)
        hidden = int(d_model * mlp_ratio)
        if mlp == "swiglu":
            self.mlp = SwiGLU(d_model, 8 * round(hidden * 2 / 3 / 8), dropout)
        else:
            assert mlp == "gelu", mlp
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model), nn.Dropout(dropout)
            )

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
        qk_norm: bool = True, mlp: str = "gelu", norm: str = "layernorm", init: str = "default",
    ):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio, dropout, qk_norm, mlp, norm) for _ in range(n_layers)]
        )
        self.ln_f = NORMS[norm](d_model)
        if init == "xavier":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            for blk in self.blocks:
                out = blk.mlp.out if mlp == "swiglu" else blk.mlp[2]
                for w in (blk.attn.proj.weight, out.weight):
                    w.data.mul_((2 * n_layers) ** -0.5)
        else:
            assert init == "default", init

    def forward(self, x: torch.Tensor, valid: torch.Tensor, causal: bool) -> torch.Tensor:
        """x: (B, N, C); valid: (B, N) bool key mask. Token 0 is always valid, so no row is fully masked."""
        N = x.shape[1]
        cos, sin = rope_cache(N, self.head_dim, x.device)
        mask = valid[:, None, None, :]
        if causal:
            mask = mask & torch.ones(N, N, dtype=torch.bool, device=x.device).tril()
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.ln_f(x)
