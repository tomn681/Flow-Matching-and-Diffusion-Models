import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from nn.ops.convolution import ConvND
from nn.ops.normalization import RMSNormND, make_group_norm
from nn.ops.upsampling import DownsampleND, UpsampleND
from .timestep import TimestepBlock
from .common import zero_module


class ResBlockND(TimestepBlock):
    """
    Residual block with optional timestep/state conditioning.

    Attributes:
        channels -> [int] Input channels.
        emb_channels -> [int | None] Embedding channels when conditioning is used.
        dropout -> [float] Dropout rate.
        out_channels -> [int, default: None] Output channels. If None, uses channels.
        use_conv -> [bool, default: False] Whether to use a convolution in the skip branch.
        spatial_dims -> [int, default: 2] Dimensionality of the convolution (1/2/3D).
    """

    def __init__(
        self,
        channels: int,
        emb_channels: Optional[int],
        dropout: float,
        out_channels: int = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        spatial_dims: int = 2,
        norm_type: str = "gn",
        act: str = "silu",
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        zero_init_last_conv: bool = True,
        emb_activation_before_proj: bool = False,
        add_embedding_to_hidden: bool = False,
        *,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        skip_time_act: Optional[bool] = None,
        time_embedding_norm: str | None = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        up: bool = False,
        down: bool = False,
        kernel: str | None = None,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        self.uses_embedding = emb_channels is not None
        self.emb_activation_before_proj = emb_activation_before_proj
        self.add_embedding_to_hidden = add_embedding_to_hidden
        self.pre_norm = bool(pre_norm)
        self.up = bool(up)
        self.down = bool(down)
        self.kernel = kernel
        self.output_scale_factor = float(output_scale_factor)
        self.conv_2d_out_channels = conv_2d_out_channels or self.out_channels
        self.time_embedding_norm = (
            str(time_embedding_norm)
            if time_embedding_norm is not None
            else ("scale_shift" if self.use_scale_shift_norm else ("default" if self.uses_embedding else "none"))
        )
        self.skip_time_act = bool(skip_time_act) if skip_time_act is not None else False

        if emb_channels is None and (use_scale_shift_norm or self.time_embedding_norm in {"default", "scale_shift"}):
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")
        if self.time_embedding_norm == "scale_shift":
            self.use_scale_shift_norm = True
            self.add_embedding_to_hidden = False
        elif self.time_embedding_norm == "default":
            self.use_scale_shift_norm = False
            self.add_embedding_to_hidden = True
        elif self.time_embedding_norm == "none":
            self.use_scale_shift_norm = False
            self.add_embedding_to_hidden = False
        else:
            raise ValueError(f"Unsupported time_embedding_norm '{self.time_embedding_norm}'")
        if emb_channels is not None and not self.use_scale_shift_norm and not self.add_embedding_to_hidden:
            raise ValueError(
                "emb_channels was provided but the residual block has no embedding-consumption path. "
                "Enable use_scale_shift_norm or add_embedding_to_hidden, or remove emb_channels."
            )
        if groups_out is None:
            groups_out = norm_groups

        self.norm1 = self._make_norm(norm_type, channels, norm_groups, norm_eps)
        self.act1 = self._make_act(act)
        self.conv1 = ConvND(spatial_dims, channels, self.out_channels, 3, padding=1)

        if self.uses_embedding:
            self.emb_act = self._make_act(act)
            self.emb_layers = nn.Linear(
                emb_channels,
                2 * self.out_channels if self.use_scale_shift_norm else self.out_channels,
            )
        else:
            self.emb_layers = None

        self.norm2 = self._make_norm(norm_type, self.out_channels, groups_out, norm_eps)
        self.act2 = self._make_act(act)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv2 = ConvND(spatial_dims, self.out_channels, self.conv_2d_out_channels, 3, padding=1)
        if zero_init_last_conv:
            self.conv2 = zero_module(self.conv2)

        self.upsample = None
        self.downsample = None
        if self.up:
            if spatial_dims != 2 and kernel not in {None, "sde_vp"}:
                raise ValueError("ND ResBlock upsampling only supports the HF default nearest-neighbor branch.")
            self.upsample = UpsampleND(spatial_dims, channels, use_conv=False)
        elif self.down:
            if spatial_dims != 2 and kernel not in {None, "sde_vp"}:
                raise ValueError("ND ResBlock downsampling only supports the HF default average-pool branch.")
            self.downsample = DownsampleND(
                spatial_dims,
                channels,
                use_conv=False,
            )

        self.use_in_shortcut = (
            (channels != self.conv_2d_out_channels) if use_in_shortcut is None else bool(use_in_shortcut)
        )
        if not self.use_in_shortcut:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = ConvND(
                spatial_dims,
                channels,
                self.conv_2d_out_channels,
                3,
                padding=1,
                bias=conv_shortcut_bias,
            )
        else:
            self.skip_connection = ConvND(
                spatial_dims,
                channels,
                self.conv_2d_out_channels,
                1,
                bias=conv_shortcut_bias,
            )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply the block to a Tensor, optionally conditioned on an embedding.

        Args:
            x -> [torch.Tensor] Features (N, C, L) | (N, C, H, W) | (N, C, D, H, W)
            emb -> [torch.Tensor | None] Conditioning embeddings (N, emb_channels)

        Returns:
            [torch.Tensor] Outputs (N, out_channels, L) | (N, out_channels, H, W) | (N, out_channels, D, H, W)
        """
        h = self.norm1(x)
        h = self.act1(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.uses_embedding:
            if emb is None:
                raise ValueError("ResBlockND expects `emb` when emb_channels is set.")
            if not self.skip_time_act:
                emb = self.emb_act(emb)
            emb_out = self.emb_layers(emb).type(h.dtype)
            # (N, 2*out_channels) if use_scale_shift_norm else (N, out_channels)
            emb_out = emb_out.view(*emb_out.shape, *([1] * (h.ndim - emb_out.ndim))) 
            # (N, 2*out_channels, 1, 1)  if use_scale_shift_norm else (N, out_channels, 1, 1)

            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
            elif self.add_embedding_to_hidden:
                h = h + emb_out
        h = self.norm2(h)
        if self.use_scale_shift_norm and self.uses_embedding:
            h = h * (1 + scale) + shift
        h = self.act2(h)
        h = self.dropout_layer(h)
        h = self.conv2(h)

        return (self.skip_connection(x) + h) / self.output_scale_factor

    @staticmethod
    def _make_norm(norm_type: str, channels: int, norm_groups: int, norm_eps: float) -> nn.Module:
        norm_type = norm_type.lower()
        if norm_type == "gn":
            return make_group_norm(channels, groups=norm_groups, eps=norm_eps)
        if norm_type == "rmsnorm":
            return RMSNormND(channels)
        raise ValueError(f"Unsupported norm_type '{norm_type}'")

    @staticmethod
    def _make_act(act: str) -> Callable[[torch.Tensor], torch.Tensor]:
        act = act.lower()
        if act == "silu" or act == "swish":
            return nn.SiLU()
        if act == "relu":
            return nn.ReLU()
        if act == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation '{act}'")


# Convenience factories for common norm/activation pairs.
def build_resblock_gn_silu(**kwargs) -> ResBlockND:
    return ResBlockND(norm_type="gn", act="silu", **kwargs)


def build_resblock_gn_swish(**kwargs) -> ResBlockND:
    return ResBlockND(norm_type="gn", act="swish", **kwargs)


def build_resblock_rmsnorm_silu(**kwargs) -> ResBlockND:
    return ResBlockND(norm_type="rmsnorm", act="silu", **kwargs)


def build_resblock_rmsnorm_swish(**kwargs) -> ResBlockND:
    return ResBlockND(norm_type="rmsnorm", act="swish", **kwargs)
