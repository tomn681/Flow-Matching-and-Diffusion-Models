from __future__ import annotations

from collections.abc import Mapping

import torch


HF_UNET_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (".query.", ".to_q."),
    (".key.", ".to_k."),
    (".value.", ".to_v."),
    (".proj_attn.", ".to_out.0."),
    (".conv1.weight", ".conv1.conv.weight"),
    (".conv1.bias", ".conv1.conv.bias"),
    (".conv2.weight", ".conv2.conv.weight"),
    (".conv2.bias", ".conv2.conv.bias"),
    (".time_emb_proj.weight", ".emb_layers.weight"),
    (".time_emb_proj.bias", ".emb_layers.bias"),
    (".conv_shortcut.weight", ".skip_connection.conv.weight"),
    (".conv_shortcut.bias", ".skip_connection.conv.bias"),
    (".downsamplers.0.conv.weight", ".downsamplers.0.op.conv.weight"),
    (".downsamplers.0.conv.bias", ".downsamplers.0.op.conv.bias"),
    (".upsamplers.0.conv.weight", ".upsamplers.0.conv.conv.weight"),
    (".upsamplers.0.conv.bias", ".upsamplers.0.conv.conv.bias"),
    (".proj_in.weight", ".proj_in.conv.weight"),
    (".proj_in.bias", ".proj_in.conv.bias"),
    (".proj_out.weight", ".proj_out.conv.weight"),
    (".proj_out.bias", ".proj_out.conv.bias"),
)

HF_VAE_KEY_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (".conv_in.weight", ".conv_in.conv.weight"),
    (".conv_in.bias", ".conv_in.conv.bias"),
    (".conv_out.weight", ".conv_out.conv.weight"),
    (".conv_out.bias", ".conv_out.conv.bias"),
    (".conv1.weight", ".conv1.conv.weight"),
    (".conv1.bias", ".conv1.conv.bias"),
    (".conv2.weight", ".conv2.conv.weight"),
    (".conv2.bias", ".conv2.conv.bias"),
    (".conv_shortcut.weight", ".skip_connection.conv.weight"),
    (".conv_shortcut.bias", ".skip_connection.conv.bias"),
    (".downsamplers.0.conv.weight", ".down.op.conv.weight"),
    (".downsamplers.0.conv.bias", ".down.op.conv.bias"),
    (".upsamplers.0.conv.weight", ".up.conv.conv.weight"),
    (".upsamplers.0.conv.bias", ".up.conv.conv.bias"),
    (".quant_conv.weight", ".quant_conv.conv.weight"),
    (".quant_conv.bias", ".quant_conv.conv.bias"),
    (".post_quant_conv.weight", ".post_quant_conv.conv.weight"),
    (".post_quant_conv.bias", ".post_quant_conv.conv.bias"),
)


def map_hf_unet_key_to_ours(key: str, key_map: Mapping[str, str] | None = None) -> str:
    """Map one HuggingFace UNet key to this repository's UNet key naming."""
    if key_map is not None and key in key_map:
        return key_map[key]
    mapped = key
    if mapped.startswith("conv1."):
        mapped = mapped.replace("conv1.", "conv1.conv.", 1)
    if mapped.startswith("conv2."):
        mapped = mapped.replace("conv2.", "conv2.conv.", 1)
    if mapped.startswith("conv_shortcut."):
        mapped = mapped.replace("conv_shortcut.", "skip_connection.conv.", 1)
    if mapped.startswith("time_emb_proj."):
        mapped = mapped.replace("time_emb_proj.", "emb_layers.", 1)
    if mapped.startswith("mid_block.attentions.0."):
        mapped = mapped.replace("mid_block.attentions.0.", "mid_block.transformer.", 1)
    for src, dst in HF_UNET_KEY_REPLACEMENTS:
        mapped = mapped.replace(src, dst)
    return mapped


def map_hf_unet_to_ours(
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    target_state_dict: Mapping[str, torch.Tensor] | None = None,
    key_map: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace UNet state dict keys to this repository's key format.

    If ``target_state_dict`` is provided, this function enforces:
    - every HF key maps to an existing target key
    - every mapped tensor shape exactly matches the target tensor shape
    """
    mapped: dict[str, torch.Tensor] = {}
    missing_target_keys: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    duplicate_mapped_keys: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        ours_key = map_hf_unet_key_to_ours(hf_key, key_map=key_map)
        if ours_key in mapped:
            duplicate_mapped_keys.append(ours_key)
            continue
        mapped[ours_key] = tensor
        if target_state_dict is None:
            continue
        target_tensor = target_state_dict.get(ours_key)
        if target_tensor is None:
            missing_target_keys.append(hf_key)
            continue
        if tuple(target_tensor.shape) != tuple(tensor.shape):
            shape_mismatches.append((hf_key, tuple(tensor.shape), tuple(target_tensor.shape)))

    if duplicate_mapped_keys:
        preview = ", ".join(sorted(set(duplicate_mapped_keys))[:8])
        raise ValueError(
            f"HF->ours key mapping produced duplicate destination keys ({len(duplicate_mapped_keys)}): {preview}"
        )
    if missing_target_keys:
        preview = ", ".join(sorted(missing_target_keys)[:8])
        raise ValueError(
            f"Unmapped HF keys for target model ({len(missing_target_keys)}). Example keys: {preview}"
        )
    if shape_mismatches:
        first = shape_mismatches[0]
        raise ValueError(
            "Shape mismatch while mapping HF UNet weights: "
            f"{first[0]} hf={first[1]} target={first[2]} "
            f"(total mismatches: {len(shape_mismatches)})"
        )
    return mapped


def map_hf_vae_key_to_ours(key: str, key_map: Mapping[str, str] | None = None) -> str:
    """Map one HuggingFace VAE key to this repository's AutoencoderKL key naming."""
    if key_map is not None and key in key_map:
        return key_map[key]
    mapped = key
    mapped = mapped.replace("encoder.down_blocks.", "encoder.downs.")
    mapped = mapped.replace("decoder.up_blocks.", "decoder.ups.")
    mapped = mapped.replace(".resnets.", ".blocks.")
    mapped = mapped.replace("encoder.conv_norm_out.", "encoder.norm_out.")
    mapped = mapped.replace("encoder.mid_block.blocks.0.", "encoder.mid_block1.")
    mapped = mapped.replace("encoder.mid_block.blocks.1.", "encoder.mid_block2.")
    mapped = mapped.replace("decoder.mid_block.blocks.0.", "decoder.mid_block1.")
    mapped = mapped.replace("decoder.mid_block.blocks.1.", "decoder.mid_block2.")
    mapped = mapped.replace("encoder.mid_block.attentions.0.group_norm.", "encoder.mid_attn.norm.")
    mapped = mapped.replace("encoder.mid_block.attentions.0.to_q.", "encoder.mid_attn.q.conv.")
    mapped = mapped.replace("encoder.mid_block.attentions.0.to_k.", "encoder.mid_attn.k.conv.")
    mapped = mapped.replace("encoder.mid_block.attentions.0.to_v.", "encoder.mid_attn.v.conv.")
    mapped = mapped.replace("encoder.mid_block.attentions.0.to_out.0.", "encoder.mid_attn.proj_out.conv.")
    mapped = mapped.replace("decoder.mid_block.attentions.0.group_norm.", "decoder.mid_attn.norm.")
    mapped = mapped.replace("decoder.mid_block.attentions.0.to_q.", "decoder.mid_attn.q.conv.")
    mapped = mapped.replace("decoder.mid_block.attentions.0.to_k.", "decoder.mid_attn.k.conv.")
    mapped = mapped.replace("decoder.mid_block.attentions.0.to_v.", "decoder.mid_attn.v.conv.")
    mapped = mapped.replace("decoder.mid_block.attentions.0.to_out.0.", "decoder.mid_attn.proj_out.conv.")
    mapped = mapped.replace("decoder.conv_norm_out.", "decoder.norm_out.")
    if mapped.startswith("quant_conv."):
        mapped = mapped.replace("quant_conv.", "quant_conv.conv.", 1)
    if mapped.startswith("post_quant_conv."):
        mapped = mapped.replace("post_quant_conv.", "post_quant_conv.conv.", 1)
    for src, dst in HF_VAE_KEY_REPLACEMENTS:
        mapped = mapped.replace(src, dst)
    return mapped


def map_hf_vae_to_ours(
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    target_state_dict: Mapping[str, torch.Tensor] | None = None,
    key_map: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace AutoencoderKL state dict keys to this repository's key format."""
    mapped: dict[str, torch.Tensor] = {}
    duplicate_mapped_keys: list[str] = []

    for hf_key, tensor in hf_state_dict.items():
        ours_key = map_hf_vae_key_to_ours(hf_key, key_map=key_map)
        if ours_key in mapped:
            duplicate_mapped_keys.append(ours_key)
            continue
        mapped[ours_key] = tensor

    mapped = _merge_qkv_for_mha(mapped)

    missing_target_keys: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    if target_state_dict is not None:
        remapped: dict[str, torch.Tensor] = {}
        for ours_key, tensor in mapped.items():
            mapped_tensor = tensor
            target_tensor = target_state_dict.get(ours_key)
            if target_tensor is None:
                missing_target_keys.append(ours_key)
                continue
            if (
                mapped_tensor.dim() == 2
                and target_tensor.dim() == 4
                and tuple(target_tensor.shape[:2]) == tuple(mapped_tensor.shape)
                and tuple(target_tensor.shape[2:]) == (1, 1)
            ):
                mapped_tensor = mapped_tensor[:, :, None, None]
            remapped[ours_key] = mapped_tensor
            if tuple(target_tensor.shape) != tuple(mapped_tensor.shape):
                shape_mismatches.append((ours_key, tuple(mapped_tensor.shape), tuple(target_tensor.shape)))
        mapped = remapped

    if duplicate_mapped_keys:
        preview = ", ".join(sorted(set(duplicate_mapped_keys))[:8])
        raise ValueError(
            f"HF->ours VAE key mapping produced duplicate destination keys ({len(duplicate_mapped_keys)}): {preview}"
        )
    if missing_target_keys:
        preview = ", ".join(sorted(missing_target_keys)[:8])
        raise ValueError(
            f"Unmapped HF VAE keys for target model ({len(missing_target_keys)}). Example keys: {preview}"
        )
    if shape_mismatches:
        first = shape_mismatches[0]
        raise ValueError(
            "Shape mismatch while mapping HF VAE weights: "
            f"{first[0]} hf={first[1]} target={first[2]} "
            f"(total mismatches: {len(shape_mismatches)})"
        )
    return mapped


def _merge_qkv_for_mha(mapped: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Merge separate HF q/k/v VAE attention projections into fused MultiheadAttention weights.

    The current SpatialSelfAttention implementation stores mid-block attention as:
    - `attn.in_proj_weight`
    - `attn.in_proj_bias`
    - `attn.out_proj.{weight,bias}`

    HF AutoencoderKL stores these as separate:
    - `to_q`
    - `to_k`
    - `to_v`
    - `to_out.0`
    """
    out = dict(mapped)
    for prefix in ("encoder.mid_attn", "decoder.mid_attn"):
        q_w_key = f"{prefix}.q.conv.weight"
        k_w_key = f"{prefix}.k.conv.weight"
        v_w_key = f"{prefix}.v.conv.weight"
        q_b_key = f"{prefix}.q.conv.bias"
        k_b_key = f"{prefix}.k.conv.bias"
        v_b_key = f"{prefix}.v.conv.bias"
        out_w_key = f"{prefix}.proj_out.conv.weight"
        out_b_key = f"{prefix}.proj_out.conv.bias"

        required = (q_w_key, k_w_key, v_w_key, q_b_key, k_b_key, v_b_key, out_w_key, out_b_key)
        if not all(key in out for key in required):
            continue

        q_w = out.pop(q_w_key)
        k_w = out.pop(k_w_key)
        v_w = out.pop(v_w_key)
        q_b = out.pop(q_b_key)
        k_b = out.pop(k_b_key)
        v_b = out.pop(v_b_key)
        out_w = out.pop(out_w_key)
        out_b = out.pop(out_b_key)

        out[f"{prefix}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        out[f"{prefix}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        out[f"{prefix}.attn.out_proj.weight"] = out_w
        out[f"{prefix}.attn.out_proj.bias"] = out_b
    return out


def load_hf_unet_weights(
    model: torch.nn.Module,
    hf_model_id: str,
    *,
    subfolder: str = "unet",
    key_map: Mapping[str, str] | None = None,
    hf_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Load HuggingFace UNet weights into this repository's UNet model.

    ``hf_state_dict`` can be provided directly for offline/unit-test usage.
    Otherwise weights are loaded from ``diffusers.UNet2DConditionModel``.
    """
    state = hf_state_dict
    if state is None:
        try:
            from diffusers import UNet2DConditionModel
        except Exception as exc:  # pragma: no cover - import availability varies by environment
            raise RuntimeError(
                "diffusers is required to download HF UNet weights. "
                "Install diffusers or pass hf_state_dict explicitly."
            ) from exc
        hf_model = UNet2DConditionModel.from_pretrained(hf_model_id, subfolder=subfolder)
        state = hf_model.state_dict()
        del hf_model

    mapped = map_hf_unet_to_ours(state, target_state_dict=model.state_dict(), key_map=key_map)
    model.load_state_dict(mapped, strict=True)


def load_hf_vae_weights(
    model: torch.nn.Module,
    hf_model_id: str,
    *,
    subfolder: str = "vae",
    key_map: Mapping[str, str] | None = None,
    hf_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Load HuggingFace AutoencoderKL weights into this repository's VAE model."""
    state = hf_state_dict
    if state is None:
        try:
            from diffusers import AutoencoderKL
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "diffusers is required to download HF VAE weights. "
                "Install diffusers or pass hf_state_dict explicitly."
            ) from exc
        hf_model = AutoencoderKL.from_pretrained(hf_model_id, subfolder=subfolder)
        state = hf_model.state_dict()
        del hf_model

    mapped = map_hf_vae_to_ours(state, target_state_dict=model.state_dict(), key_map=key_map)
    model.load_state_dict(mapped, strict=True)
