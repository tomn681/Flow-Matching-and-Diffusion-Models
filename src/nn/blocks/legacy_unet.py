from __future__ import annotations

import torch
import torch.nn as nn

from nn.blocks.attention import DiffusersAttentionND
from nn.blocks.residual import ResBlockND
from nn.ops.upsampling import DownsampleND, UpsampleND


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
                    DiffusersAttentionND(out_channels, heads=heads, eps=eps, norm_num_groups=groups)
                )
            ch = out_channels
        self.downsamplers = (
            nn.ModuleList([DownsampleND(spatial_dims, out_channels, use_conv=True)])
            if add_downsample
            else None
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        output_states = ()
        for idx, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[idx](hidden_states)
            output_states = output_states + (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


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
                    DiffusersAttentionND(out_channels, heads=heads, eps=eps, norm_num_groups=groups)
                )
        self.upsamplers = (
            nn.ModuleList([UpsampleND(spatial_dims, out_channels, use_conv=True)])
            if add_upsample
            else None
        )

    def forward(self, hidden_states: torch.Tensor, res_hidden_states_tuple, temb: torch.Tensor):
        for idx, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if self.attentions is not None:
                hidden_states = self.attentions[idx](hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


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
            nn.ModuleList([DiffusersAttentionND(in_channels, heads=heads, eps=eps, norm_num_groups=groups)])
            if add_attention
            else None
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        hidden_states = self.resnets[0](hidden_states, temb)
        if self.attentions is not None:
            hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states
