import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import zero_module

class QKVAttention(nn.Module):
    """
    Implementation of Scaled Dot Product Attention.
    
    Attributes:
        - efficient_attn [bool, default: True] Uses the efficient PyTorch implementation.
        - dropout [float, default: 0.0] Dropout value.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        has_sdp = hasattr(F, "scaled_dot_product_attention")
        self.efficient_attn = bool(efficient_attn and has_sdp)
        if efficient_attn and not has_sdp:
            warnings.warn(
                "Efficient scaled dot-product attention requires PyTorch >= 2.0. "
                "Falling back to the explicit implementation.",
                RuntimeWarning,
                stacklevel=2,
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q -> [torch.Tensor] Queries (N, ..., Q_heads, Tgt_dim, QK_emb)
            k -> [torch.Tensor] Keys (N, ..., KV_heads, S_dim, QK_emb)
            v -> [torch.Tensor] Values (N, ..., KV_heads, S_dim, V_emb)
            S_dim stands for source dimension.
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=False
            )

        scale = 1 / math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.matmul(attn, v)


class LinearQKVAttention(nn.Module):
    """
    Linear attention variant that factors the softmax to reduce memory usage.
    """

    def __init__(self, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.dropout = dropout
        self.eps = eps

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_softmax = F.softmax(k, dim=-2)
        q_softmax = F.softmax(q, dim=-1)

        context = torch.einsum("...nd,...ne->...de", k_softmax, v)
        context = context / (k_softmax.sum(dim=-2, keepdim=False).unsqueeze(-1) + self.eps)
        out = torch.einsum("...nd,...de->...ne", q_softmax, context)
        return F.dropout(out, p=self.dropout, training=self.training)


class ContextBlock(nn.Module):
    """
    Base class for layers that consume an external context tensor.
    """

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class SpatialSelfAttention(nn.Module):
    """
    Implementation of Multi-head Spatial Self-Attention Block.
    
    Ported from CompVis fm-boosting
    https://github.com/CompVis/fm-boosting/tree/main
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64,
                 use_linear: bool = False, use_efficient_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(max(1, math.gcd(dim, 32)), dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.dim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (b, c, *spatial), where spatial can be (f, h, w) or (h, w).
        Returns:
            x: Tensor after attention, MHSA(x) + residual.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)                                     # (b, c, f * h * w)
        qkv = self.qkv(self.norm(x))                                # (b, 3 * c * nh, f * h * w)
        qkv = qkv.reshape(b, self.heads, qkv.shape[-1], -1)         # (b, nh, f * h * w, 3 * c)
        q, k, v = qkv.chunk(3, dim=-1)                              # (b, nh, f * h * w, c) each
        h = self.attention(q, k, v)                                 # (b, nh, f * h * w, c)
        h = h.reshape(b, self.inner_dim, -1)                        # (b, nh * c, f * h * w)
        h = self.proj_out(h)                                        # (b, c, f * h * w)
        return (x + h).reshape(b, c, *spatial)


class SpatialCrossAttention(ContextBlock):
    """
    Multi-head spatial cross-attention block that attends `x` to `context`.

    Expected shapes:
        - x: (b, c, *spatial)
        - context: (b, c_ctx, *spatial_ctx) or (b, tokens, c_ctx)

    Notes:
        - The context is flattened into tokens; spatial dims can differ from x.
        - Default context channels are set by the caller (e.g., VAE latent channels).
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 4,
        dim_head: int = 64,
        use_linear: bool = False,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(max(1, math.gcd(dim, 32)), dim)
        self.context_norm = nn.GroupNorm(max(1, math.gcd(context_dim, 32)), context_dim)
        self.q_proj = nn.Conv1d(dim, self.inner_dim, 1)
        self.kv_proj = nn.Conv1d(context_dim, self.inner_dim * 2, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.dim, 1))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context is None:
            raise ValueError("SpatialCrossAttention requires a non-empty context tensor.")

        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)  # (b, c, tokens_x)

        if context.dim() == 3:
            if context.shape[1] == self.context_dim:
                ctx_flat = context
            elif context.shape[-1] == self.context_dim:
                ctx_flat = context.transpose(1, 2)
            else:
                raise ValueError(
                    f"Context channels mismatch: expected {self.context_dim}, got {context.shape}."
                )
        else:
            if context.shape[1] != self.context_dim:
                raise ValueError(
                    f"Context channels mismatch: expected {self.context_dim}, got {context.shape}."
                )
            ctx_flat = context.reshape(context.shape[0], context.shape[1], -1)

        q = self.q_proj(self.norm(x_flat))
        kv = self.kv_proj(self.context_norm(ctx_flat))

        q = q.reshape(b, self.heads, q.shape[-1], -1)
        kv = kv.reshape(b, self.heads, kv.shape[-1], -1)
        k, v = kv.chunk(2, dim=-1)

        h = self.attention(q, k, v)
        h = h.reshape(b, self.inner_dim, -1)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


class DiffusersAttentionND(nn.Module):
    """
    Diffusers-style self-attention over flattened spatial tokens.

    Keeps explicit projection modules (to_q/to_k/to_v/to_out) useful for
    state-dict conversion with Diffusers-like checkpoints.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        context_dim: int | None = None,
        norm_num_groups: int = 32,
        eps: float = 1e-5,
        dropout: float = 0.0,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.heads = max(1, heads)
        self.head_dim = channels // self.heads
        self.context_dim = int(context_dim) if context_dim is not None else None
        self.group_norm = nn.GroupNorm(max(1, math.gcd(channels, norm_num_groups)), channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        if self.context_dim is None:
            self.context_norm = None
            self.to_k = nn.Linear(channels, channels)
            self.to_v = nn.Linear(channels, channels)
        else:
            self.context_norm = nn.GroupNorm(
                max(1, math.gcd(self.context_dim, norm_num_groups)),
                self.context_dim,
                eps=eps,
            )
            self.to_k = nn.Linear(self.context_dim, channels)
            self.to_v = nn.Linear(self.context_dim, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(dropout)])
        self.attention = QKVAttention(efficient_attn=use_efficient_attn, dropout=dropout)

    def forward(self, hidden_states: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, c = hidden_states.shape[:2]
        spatial = hidden_states.shape[2:]
        x = hidden_states.reshape(b, c, -1)
        x = self.group_norm(x).transpose(1, 2)  # [B, T, C]

        q = self.to_q(x)
        if self.context_dim is None:
            kv_source = x
        else:
            if context is None:
                raise ValueError("DiffusersAttentionND cross-attention requires a non-empty context tensor.")
            if context.dim() == 3:
                if context.shape[1] == self.context_dim:
                    ctx = context
                elif context.shape[-1] == self.context_dim:
                    ctx = context.transpose(1, 2)
                else:
                    raise ValueError(
                        f"Context channels mismatch: expected {self.context_dim}, got {tuple(context.shape)}."
                    )
            else:
                if context.shape[1] != self.context_dim:
                    raise ValueError(
                        f"Context channels mismatch: expected {self.context_dim}, got {tuple(context.shape)}."
                    )
                ctx = context.reshape(context.shape[0], context.shape[1], -1)
            ctx = self.context_norm(ctx).transpose(1, 2)  # [B, T_ctx, C_ctx]
            kv_source = ctx

        k = self.to_k(kv_source)
        v = self.to_v(kv_source)

        q = q.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        out = self.attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, -1, c)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        out = out.transpose(1, 2).reshape(b, c, *spatial)
        return out + hidden_states


def run_self_tests() -> None:
    torch.manual_seed(0)

    def _test_qkv_attention():
        batch, heads, tokens, dim = 2, 4, 8, 16
        q = torch.randn(batch, heads, tokens, dim)
        k = torch.randn(batch, heads, tokens, dim)
        v = torch.randn(batch, heads, tokens, dim)

        vanilla = QKVAttention(efficient_attn=False, dropout=0.0)
        vanilla_out = vanilla(q, k, v)
        assert vanilla_out.shape == q.shape, "Vanilla attention shape mismatch"

        efficient = QKVAttention(efficient_attn=True, dropout=0.0)
        efficient.eval()
        efficient_out = efficient(q, k, v)
        assert efficient_out.shape == q.shape, "Efficient attention shape mismatch"
        print("QKVAttention variants passed.")

        linear = LinearQKVAttention(dropout=0.0)
        linear_out = linear(q, k, v)
        assert linear_out.shape == q.shape, "Linear attention shape mismatch"
        print("LinearQKVAttention passed.")

    def _test_spatial_self_attention():
        configs = [
            dict(channels=32, spatial=(8, 8), use_linear=False),
            dict(channels=32, spatial=(8, 8), use_linear=True),
            dict(channels=16, spatial=(4, 8, 8), use_linear=False),
            dict(channels=16, spatial=(4, 8, 8), use_linear=True),
        ]

        for cfg in configs:
            shape = (1, cfg["channels"], *cfg["spatial"])
            x = torch.randn(shape)
            block = SpatialSelfAttention(
                dim=cfg["channels"],
                heads=4,
                dim_head=cfg["channels"] // 4,
                use_linear=cfg["use_linear"],
                use_efficient_attn=True,
            )
            y = block(x)
            assert y.shape == x.shape, f"SpatialSelfAttention failed for {cfg}"
        print("SpatialSelfAttention variants passed.")

    def _test_spatial_cross_attention():
        x = torch.randn((2, 16, 8, 8))
        ctx = torch.randn((2, 4, 8, 8))
        block = SpatialCrossAttention(
            dim=16,
            context_dim=4,
            heads=4,
            dim_head=4,
            use_linear=False,
            use_efficient_attn=True,
        )
        y = block(x, ctx)
        assert y.shape == x.shape, "SpatialCrossAttention output shape mismatch"
        print("SpatialCrossAttention passed.")

    _test_qkv_attention()
    _test_spatial_self_attention()
    _test_spatial_cross_attention()
    print("All attention module self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
