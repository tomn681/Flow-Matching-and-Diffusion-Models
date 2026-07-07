from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.ops.convolution import ConvND
from nn.ops.normalization import make_group_norm


class FeedForward(nn.Module):
    class GEGLU(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.proj = nn.Linear(in_dim, out_dim * 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            value, gate = self.proj(x).chunk(2, dim=-1)
            return value * F.gelu(gate)

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            self.GEGLU(dim, inner),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """
    Transformer block: self-attn -> cross-attn -> FFN.

    Inputs/outputs are token tensors shaped [B, T, C].
    """

    class DiffusersCrossAttention(nn.Module):
        def __init__(
            self,
            query_dim: int,
            heads: int,
            dim_head: int,
            dropout: float = 0.0,
            cross_attention_dim: int | None = None,
        ) -> None:
            super().__init__()
            self.heads = int(heads)
            self.dim_head = int(dim_head)
            self.inner_dim = self.heads * self.dim_head
            self.scale = self.dim_head ** -0.5
            context_dim = int(cross_attention_dim or query_dim)
            self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=True), nn.Dropout(dropout)])

        def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
            b, t, _ = x.shape
            return x.view(b, t, self.heads, self.dim_head).transpose(1, 2)

        def forward(
            self,
            hidden_states: torch.Tensor,
            context: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            context = hidden_states if context is None else context
            q = self._reshape_heads(self.to_q(hidden_states))
            k = self._reshape_heads(self.to_k(context))
            v = self._reshape_heads(self.to_v(context))
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                mask = key_padding_mask.to(torch.bool)[:, None, None, :]
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        if inner_dim != dim:
            # keep exact hidden size contract for predictable parity
            raise ValueError(
                f"BasicTransformerBlock expects dim == heads*head_dim, got {dim} != {inner_dim}."
            )

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = self.DiffusersCrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = self.DiffusersCrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # self-attention
        residual = hidden_states
        x = self.norm1(hidden_states)
        x = self.attn1(x, context=None, key_padding_mask=attention_mask)
        hidden_states = residual + x

        # cross-attention
        residual = hidden_states
        x = self.norm2(hidden_states)
        if encoder_hidden_states is None:
            context = x
            key_padding_mask = attention_mask
        else:
            context = encoder_hidden_states
            key_padding_mask = encoder_attention_mask
        x = self.attn2(x, context=context, key_padding_mask=key_padding_mask)
        hidden_states = residual + x

        # feed-forward
        hidden_states = hidden_states + self.ff(self.norm3(hidden_states))
        return hidden_states


class Transformer2DModelND(nn.Module):
    """
    Diffusers Transformer2DModel-style block generalized to ND tensors.

    Pipeline: norm -> proj_in -> flatten -> transformer blocks -> unflatten -> proj_out + residual.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.in_channels = int(in_channels)
        self.inner_dim = int(num_attention_heads * attention_head_dim)
        if self.inner_dim <= 0:
            raise ValueError("inner transformer dim must be > 0")

        self.norm = make_group_norm(in_channels, groups=norm_num_groups, eps=1e-6)
        self.proj_in = ConvND(self.spatial_dims, in_channels, self.inner_dim, kernel_size=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = ConvND(self.spatial_dims, self.inner_dim, in_channels, kernel_size=1, padding=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.norm(hidden_states)
        x = self.proj_in(x)

        b, c = x.shape[:2]
        spatial = x.shape[2:]
        x = x.reshape(b, c, -1).transpose(1, 2)  # [B, T, C]

        for block in self.transformer_blocks:
            x = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        x = x.transpose(1, 2).reshape(b, c, *spatial)
        x = self.proj_out(x)
        return x + residual


def build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("Position embedding dim must be > 0.")
    if dim % 4 != 0:
        raise ValueError(f"2D sin/cos position embedding requires dim divisible by 4, got {dim}.")

    quarter_dim = dim // 4
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(quarter_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter_dim, 1)))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return pos.unsqueeze(0).to(dtype=dtype)


class BottleneckTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"BottleneckTransformerLayer requires hidden_size ({hidden_size}) divisible by num_heads ({num_heads})."
            )
        mlp_hidden = max(hidden_size, int(round(hidden_size * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class TransformerBottleneck2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        positional_embedding_type: str = "2d_sincos",
        attention_impl: str = "mha",
        zero_init_proj_out: bool = True,
    ) -> None:
        super().__init__()
        if attention_impl != "mha":
            raise ValueError(
                f"Unsupported transformer bottleneck attention_impl '{attention_impl}'. Supported: mha."
            )
        if positional_embedding_type != "2d_sincos":
            raise ValueError(
                "Unsupported transformer bottleneck positional_embedding_type "
                f"'{positional_embedding_type}'. Supported: 2d_sincos."
            )

        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.norm = make_group_norm(in_channels, groups=norm_num_groups, eps=1e-6)
        self.proj_in = ConvND(2, in_channels, self.hidden_size, kernel_size=1, padding=0)
        self.layers = nn.ModuleList(
            [
                BottleneckTransformerLayer(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.proj_out = ConvND(2, self.hidden_size, in_channels, kernel_size=1, padding=0)
        if zero_init_proj_out:
            nn.init.zeros_(self.proj_out.conv.weight)
            if self.proj_out.conv.bias is not None:
                nn.init.zeros_(self.proj_out.conv.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(
                f"TransformerBottleneck2D expects [B, C, H, W], got {tuple(hidden_states.shape)}."
            )
        residual = hidden_states
        x = self.norm(hidden_states)
        x = self.proj_in(x)

        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = x + build_2d_sincos_position_embedding(h, w, c, device=x.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.proj_out(x)
        return residual + x
