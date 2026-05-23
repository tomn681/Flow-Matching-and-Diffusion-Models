from __future__ import annotations

import torch
import torch.nn as nn

from nn.ops.convolution import ConvND
from nn.ops.normalization import make_group_norm


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
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
        self.attn1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention_dim = cross_attention_dim
        if cross_attention_dim is not None and cross_attention_dim != dim:
            self.encoder_proj = nn.Linear(cross_attention_dim, dim)
        else:
            self.encoder_proj = None
        self.attn2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
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
        x, _ = self.attn1(x, x, x, key_padding_mask=attention_mask)
        hidden_states = residual + x

        # cross-attention
        residual = hidden_states
        x = self.norm2(hidden_states)
        if encoder_hidden_states is None:
            kv = x
            key_padding_mask = attention_mask
        else:
            kv = encoder_hidden_states
            if self.encoder_proj is not None:
                kv = self.encoder_proj(kv)
            key_padding_mask = encoder_attention_mask
        x, _ = self.attn2(x, kv, kv, key_padding_mask=key_padding_mask)
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
