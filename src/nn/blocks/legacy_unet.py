from __future__ import annotations

import torch
import torch.nn as nn

from core.registry import Registry
from nn.blocks.attention import DiffusersAttentionND
from nn.blocks.residual import ResBlockND
from nn.blocks.transformer import Transformer2DModelND
from nn.ops.upsampling import DownsampleND, UpsampleND


BLOCK_REGISTRY = Registry[nn.Module]("unet_blocks", base_type=nn.Module)


@BLOCK_REGISTRY.register("DownBlock2D")
class DownBlock2DCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        with_attention: bool = False,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        ch = in_channels
        heads = max(1, out_channels // max(attention_head_dim, 1))
        for _ in range(num_layers):
            self.resnets.append(
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=ch,
                    emb_channels=temb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                )
            )
            if with_attention:
                self.attentions.append(
                    DiffusersAttentionND(
                        out_channels,
                        heads=heads,
                        context_dim=cross_attention_dim,
                        eps=eps,
                        norm_num_groups=groups,
                    )
                )
            ch = out_channels
        self.downsamplers = (
            nn.ModuleList([DownsampleND(spatial_dims, out_channels, use_conv=True)]) if add_downsample else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        output_states = ()
        for idx, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[idx](hidden_states, context=context)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


@BLOCK_REGISTRY.register("AttnDownBlock2D")
class AttnDownBlock2DCompat(DownBlock2DCompat):
    def __init__(self, *args, **kwargs):
        kwargs["with_attention"] = True
        super().__init__(*args, **kwargs)


@BLOCK_REGISTRY.register("CrossAttnDownBlock2D")
class CrossAttnDownBlock2DCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        with_attention: bool = True,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        ch = in_channels
        heads = max(1, out_channels // max(attention_head_dim, 1))
        for _ in range(num_layers):
            self.resnets.append(
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=ch,
                    emb_channels=temb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                )
            )
            self.attentions.append(
                Transformer2DModelND(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=heads,
                    attention_head_dim=max(1, out_channels // heads),
                    num_layers=transformer_layers_per_block,
                    dropout=dropout,
                    norm_num_groups=groups,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            ch = out_channels
        self.downsamplers = (
            nn.ModuleList([DownsampleND(spatial_dims, out_channels, use_conv=True)]) if add_downsample else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        output_states = ()
        for idx, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = self.attentions[idx](
                hidden_states,
                encoder_hidden_states=context,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


@BLOCK_REGISTRY.register("UpBlock2D")
class UpBlock2DCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        add_upsample: bool,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        with_attention: bool = False,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        heads = max(1, out_channels // max(attention_head_dim, 1))
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=resnet_in_channels + res_skip_channels,
                    emb_channels=temb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                )
            )
            if with_attention:
                self.attentions.append(
                    DiffusersAttentionND(
                        out_channels,
                        heads=heads,
                        context_dim=cross_attention_dim,
                        eps=eps,
                        norm_num_groups=groups,
                    )
                )
        self.upsamplers = nn.ModuleList([UpsampleND(spatial_dims, out_channels, use_conv=True)]) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        for idx, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[idx](hidden_states, context=context)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


@BLOCK_REGISTRY.register("AttnUpBlock2D")
class AttnUpBlock2DCompat(UpBlock2DCompat):
    def __init__(self, *args, **kwargs):
        kwargs["with_attention"] = True
        super().__init__(*args, **kwargs)


@BLOCK_REGISTRY.register("CrossAttnUpBlock2D")
class CrossAttnUpBlock2DCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        add_upsample: bool,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        with_attention: bool = True,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        heads = max(1, out_channels // max(attention_head_dim, 1))
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=resnet_in_channels + res_skip_channels,
                    emb_channels=temb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                )
            )
            self.attentions.append(
                Transformer2DModelND(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=heads,
                    attention_head_dim=max(1, out_channels // heads),
                    num_layers=transformer_layers_per_block,
                    dropout=dropout,
                    norm_num_groups=groups,
                    cross_attention_dim=cross_attention_dim,
                )
            )
        self.upsamplers = nn.ModuleList([UpsampleND(spatial_dims, out_channels, use_conv=True)]) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        for idx, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = self.attentions[idx](
                hidden_states,
                encoder_hidden_states=context,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


@BLOCK_REGISTRY.register("UNetMidBlock2D")
class UNetMidBlock2DCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        add_attention: bool = True,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        heads = max(1, in_channels // max(attention_head_dim, 1))
        self.resnets = nn.ModuleList(
            [
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=in_channels,
                    emb_channels=temb_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                ),
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=in_channels,
                    emb_channels=temb_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                ),
            ]
        )
        self.attentions = (
            nn.ModuleList(
                [
                    DiffusersAttentionND(
                        in_channels,
                        heads=heads,
                        context_dim=cross_attention_dim,
                        eps=eps,
                        norm_num_groups=groups,
                    )
                ]
            )
            if add_attention
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        if self.attentions is not None:
            hidden_states = self.attentions[0](hidden_states, context=context)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


@BLOCK_REGISTRY.register("UNetMidBlock2DCrossAttn")
class UNetMidBlock2DCrossAttnCompat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        eps: float,
        groups: int,
        dropout: float,
        time_scale_shift: str,
        add_attention: bool = True,
        attention_head_dim: int = 8,
        cross_attention_dim: int | None = None,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        heads = max(1, in_channels // max(attention_head_dim, 1))
        self.resnets = nn.ModuleList(
            [
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=in_channels,
                    emb_channels=temb_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                ),
                ResBlockND(
                    spatial_dims=spatial_dims,
                    channels=in_channels,
                    emb_channels=temb_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    use_conv=False,
                    use_scale_shift_norm=(time_scale_shift == "scale_shift"),
                    norm_type="gn",
                    norm_groups=groups,
                    norm_eps=eps,
                    zero_init_last_conv=False,
                    emb_activation_before_proj=True,
                    add_embedding_to_hidden=True,
                ),
            ]
        )
        self.transformer = Transformer2DModelND(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=heads,
            attention_head_dim=max(1, in_channels // heads),
            num_layers=transformer_layers_per_block,
            dropout=dropout,
            norm_num_groups=groups,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.transformer(
            hidden_states,
            encoder_hidden_states=context,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states
