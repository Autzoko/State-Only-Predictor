"""Pre-LN Transformer with rotary position embeddings over patch tokens."""

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
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, cos, sin, attn_mask):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

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
