from __future__ import annotations

import torch.nn as nn

from nn.blocks.attention import DiffusersAttentionND, LegacyQKVSpatialSelfAttention, SpatialSelfAttention


def build_vae_attention_layer(
    *,
    channels: int,
    attention_impl: str,
    spatial_dims: int,
    norm_eps: float,
    zero_init_attn_out: bool,
    attn_heads: int | None,
    attn_dim_head: int | None,
) -> nn.Module:
    impl = str(attention_impl).lower()
    if impl in {"compvis", "spatial", "separate_qkv", "split_qkv"}:
        return SpatialSelfAttention(
            channels=channels,
            num_heads=attn_heads if attn_heads is not None else 1,
            spatial_dims=spatial_dims,
            norm_eps=norm_eps,
            zero_init_proj_out=zero_init_attn_out,
        )
    if impl in {"diffusers", "hf_diffusers"}:
        return DiffusersAttentionND(
            channels=channels,
            heads=attn_heads if attn_heads is not None else 1,
            context_dim=None,
            eps=norm_eps,
            use_efficient_attn=True,
        )
    if impl in {"legacy_qkv_linear", "legacy_linear_qkv", "qkv_linear", "linear_qkv", "linear"}:
        use_linear = True
    elif impl in {"legacy_qkv", "legacy_qkv_standard", "qkv", "qkv_standard", "standard"}:
        use_linear = False
    else:
        raise ValueError(
            f"Unknown attention_impl '{attention_impl}'. "
            "Expected one of: compvis, diffusers, legacy_qkv, legacy_qkv_linear."
        )

    heads = attn_heads if attn_heads is not None else 1
    if attn_dim_head is not None:
        dim_head = attn_dim_head
    elif heads == 1:
        dim_head = channels
    else:
        dim_head = max(1, channels // heads)
    return LegacyQKVSpatialSelfAttention(
        dim=channels,
        heads=heads,
        dim_head=dim_head,
        use_linear=use_linear,
        use_efficient_attn=True,
    )
