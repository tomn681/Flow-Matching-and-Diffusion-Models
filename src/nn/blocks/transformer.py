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
