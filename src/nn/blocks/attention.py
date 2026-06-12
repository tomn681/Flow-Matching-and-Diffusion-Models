import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import zero_module
from nn.ops.normalization import make_group_norm
from nn.ops.convolution import ConvND

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False):
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
                q, k, v, dropout_p=self.dropout, is_causal=bool(is_causal)
            )

        scale = 1 / math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            tgt = q.shape[-2]
            src = k.shape[-2]
            mask = torch.ones((tgt, src), device=attn.device, dtype=torch.bool).triu(diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
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


class LegacyQKVSpatialSelfAttention(nn.Module):
    """
    Legacy multi-head spatial self-attention with fused QKV projection over flattened
    tokens.

    This block preserves the historical reshape bug used by old checkpoints. Keep it
    only for compatibility with already-trained artifacts that depended on the broken
    math. New training should use `QKVSpatialSelfAttention`.
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


class QKVSpatialSelfAttention(nn.Module):
    """
    Correct fused-QKV spatial self-attention over flattened tokens.

    Expected shapes:
        - x: (b, c, *spatial)

    Notes:
        - This preserves the parameterization of the historical block
          (`norm -> Conv1d(qkv) -> attention -> zero-init Conv1d(proj_out)`),
          while fixing the head/token layout.
        - It is intended as the canonical fused-QKV attention implementation.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        use_linear: bool = False,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(max(1, math.gcd(dim, 32)), dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        residual = x.reshape(b, c, -1)
        h = self.norm(x).reshape(b, c, -1)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.heads, self.dim_head, -1).transpose(-2, -1)
        k = k.reshape(b, self.heads, self.dim_head, -1).transpose(-2, -1)
        v = v.reshape(b, self.heads, self.dim_head, -1).transpose(-2, -1)

        h = self.attention(q, k, v)
        h = h.transpose(-2, -1).reshape(b, self.inner_dim, -1)
        h = self.proj_out(h)
        return (residual + h).reshape(b, c, *spatial)


class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention for ND feature maps.

    The block normalizes `(B, C, *spatial)` inputs, flattens spatial positions
    into a token sequence, applies multi-head self-attention across those
    tokens, then restores the original spatial layout and adds a residual
    connection.

    `num_heads=1` preserves the historical single-head behavior used by older
    VAE configs, while higher values expose real multi-head control for
    `attention_impl="spatial"`.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 1,
        spatial_dims: int = 2,
        norm_eps: float = 1e-6,
        zero_init_proj_out: bool = False,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"SpatialSelfAttention requires channels ({channels}) to be divisible "
                f"by num_heads ({num_heads})."
            )
        self.channels = channels
        self.num_heads = int(num_heads)
        self.spatial_dims = int(spatial_dims)
        self.norm = make_group_norm(channels, groups=32, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=self.num_heads,
            batch_first=True,
        )
        if zero_init_proj_out:
            nn.init.zeros_(self.attn.out_proj.weight)
            if self.attn.out_proj.bias is not None:
                nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_ndim = 2 + self.spatial_dims
        if x.ndim != expected_ndim:
            raise ValueError(
                f"SpatialSelfAttention expects {self.spatial_dims}D feature maps "
                f"(N,C,*spatial), got shape {tuple(x.shape)}."
            )
        b, c, *spatial = x.shape
        h = self.norm(x).reshape(b, c, -1).transpose(1, 2)  # [B, T, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.transpose(1, 2).reshape(b, c, *spatial)
        return x + h


class LegacyQKVSpatialCrossAttention(ContextBlock):
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
        spatial_dims: int = 2,
        heads: int = 4,
        dim_head: int = 64,
        use_linear: bool = False,
        use_efficient_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_dims = int(spatial_dims)
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = make_group_norm(dim, groups=32, eps=1e-6)
        self.context_norm = make_group_norm(context_dim, groups=32, eps=1e-6)
        self.q_proj = ConvND(self.spatial_dims, dim, self.inner_dim, 1, padding=0)
        self.kv_proj = ConvND(self.spatial_dims, context_dim, self.inner_dim * 2, 1, padding=0)
        self.k_token = nn.Linear(context_dim, self.inner_dim)
        self.v_token = nn.Linear(context_dim, self.inner_dim)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(ConvND(self.spatial_dims, self.inner_dim, self.dim, 1, padding=0))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context is None:
            raise ValueError("SpatialCrossAttention requires a non-empty context tensor.")

        expected_ndim = 2 + self.spatial_dims
        if x.ndim != expected_ndim:
            raise ValueError(
                f"SpatialCrossAttention expects {self.spatial_dims}D feature maps (N,C,*spatial), got {tuple(x.shape)}."
            )
        b = x.shape[0]
        spatial = x.shape[2:]
        x_norm = self.norm(x)
        q = self.q_proj(x_norm).reshape(b, self.inner_dim, -1).transpose(1, 2)  # [B, T, Cq]

        if context.dim() == 3:
            if context.shape[1] == self.context_dim:
                ctx_tokens = context.transpose(1, 2)
            elif context.shape[-1] == self.context_dim:
                ctx_tokens = context
            else:
                raise ValueError(
                    f"Context channels mismatch: expected {self.context_dim}, got {context.shape}."
                )
            k = self.k_token(ctx_tokens)
            v = self.v_token(ctx_tokens)
        else:
            if context.ndim != expected_ndim:
                raise ValueError(
                    f"SpatialCrossAttention context must be token tensor [B,T,C] or {self.spatial_dims}D map [B,C,*spatial], "
                    f"got {tuple(context.shape)}."
                )
            if context.shape[1] != self.context_dim:
                raise ValueError(
                    f"Context channels mismatch: expected {self.context_dim}, got {context.shape}."
                )
            ctx = self.context_norm(context)
            kv = self.kv_proj(ctx).reshape(context.shape[0], self.inner_dim * 2, -1).transpose(1, 2)
            k, v = kv.chunk(2, dim=-1)

        q = q.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)

        attn_out = self.attention(q, k, v)
        h_out = attn_out.transpose(1, 2).reshape(b, self.inner_dim, *spatial)
        h_out = self.proj_out(h_out)
        return x + h_out


class SpatialCrossAttention(ContextBlock):
    """
    CompVis-style spatial cross-attention with separate q/k/v projections.

    Query projection comes from x via ConvND(1x1). Keys/values come from context:
      - map context [B, Cctx, *spatial]: ConvND 1x1 projections
      - token context [B, T, Cctx] or [B, Cctx, T]: Linear projections
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_dims: int = 2,
        norm_eps: float = 1e-6,
        zero_init_proj_out: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.spatial_dims = int(spatial_dims)
        self.norm = make_group_norm(self.dim, groups=32, eps=norm_eps)
        self.context_norm = make_group_norm(self.context_dim, groups=32, eps=norm_eps)
        self.q_proj = ConvND(self.spatial_dims, self.dim, self.dim, 1, padding=0)
        self.k_proj = ConvND(self.spatial_dims, self.context_dim, self.dim, 1, padding=0)
        self.v_proj = ConvND(self.spatial_dims, self.context_dim, self.dim, 1, padding=0)
        self.k_token = nn.Linear(self.context_dim, self.dim)
        self.v_token = nn.Linear(self.context_dim, self.dim)
        proj = ConvND(self.spatial_dims, self.dim, self.dim, 1, padding=0)
        self.proj_out = zero_module(proj) if zero_init_proj_out else proj

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context is None:
            raise ValueError("SpatialCrossAttention requires a non-empty context tensor.")
        expected_ndim = 2 + self.spatial_dims
        if x.ndim != expected_ndim:
            raise ValueError(
                f"SpatialCrossAttention expects {self.spatial_dims}D feature maps (N,C,*spatial), got {tuple(x.shape)}."
            )

        b, c = x.shape[:2]
        q = self.q_proj(self.norm(x)).reshape(b, c, -1).permute(0, 2, 1)  # [B, Tq, C]

        if context.ndim == 3:
            if context.shape[-1] == self.context_dim:
                ctx_tokens = context
            elif context.shape[1] == self.context_dim:
                ctx_tokens = context.transpose(1, 2)
            else:
                raise ValueError(
                    f"SpatialCrossAttention token context must include context_dim={self.context_dim}, got {tuple(context.shape)}."
                )
            k = self.k_token(ctx_tokens).transpose(1, 2)  # [B, C, Tk]
            v = self.v_token(ctx_tokens)                  # [B, Tk, C]
        elif context.ndim == expected_ndim:
            if context.shape[1] != self.context_dim:
                raise ValueError(
                    f"SpatialCrossAttention context channels mismatch: expected {self.context_dim}, got {tuple(context.shape)}."
                )
            ctx = self.context_norm(context)
            k = self.k_proj(ctx).reshape(b, c, -1)                      # [B, C, Tk]
            v = self.v_proj(ctx).reshape(b, c, -1).permute(0, 2, 1)     # [B, Tk, C]
        else:
            raise ValueError(
                f"SpatialCrossAttention context must be [B,T,C] tokens or [B,C,*spatial] map, got {tuple(context.shape)}."
            )

        attn = torch.bmm(q, k) * (c ** -0.5)  # [B, Tq, Tk]
        attn = torch.softmax(attn, dim=-1)
        h = torch.bmm(attn, v)                # [B, Tq, C]
        h = h.permute(0, 2, 1).reshape_as(x)
        h = self.proj_out(h)
        return x + h


# Backward-compatible alias for old imports.
ConvSpatialSelfAttention = SpatialSelfAttention


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
