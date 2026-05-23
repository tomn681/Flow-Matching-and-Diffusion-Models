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
