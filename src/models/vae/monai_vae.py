"""
MONAI-style KL VAE integrated into the existing autoencoder framework.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.types import ModelOutput
from nn.modules.vae import Decoder, Encoder
from nn.ops.convolution import ConvND
from ..registry import MODEL_REGISTRY
from .base import BaseVAE


class SpectralNormDiscriminator(nn.Module):
    """Patch discriminator that uses spectral norm instead of BatchNorm."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        spatial_dims: int = 2,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2, or 3")

        conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]

        def sn_conv(in_ch: int, out_ch: int, kernel: int, stride: int, pad: int) -> nn.Module:
            return nn.utils.spectral_norm(conv_cls(in_ch, out_ch, kernel, stride, pad))

        channels = base_channels
        layers: list[nn.Module] = [
            sn_conv(in_channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(1, n_layers):
            in_ch = channels
            channels = min(channels * 2, 512)
            layers.extend(
                [
                    sn_conv(in_ch, channels, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        penultimate = channels * 2 if n_layers >= 3 else channels
        layers.extend(
            [
                sn_conv(channels, penultimate, 4, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                sn_conv(penultimate, 1, 3, 1, 1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DiagonalGaussianSeparate:
    """Diagonal Gaussian with separate mu and sigma tensors."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, log_sigma: torch.Tensor) -> None:
        self.mu = mu
        self.std = sigma
        self.var = sigma ** 2
        self.logvar = log_sigma * 2.0
        self.device = mu.device

    def sample(self) -> torch.Tensor:
        return self.mu + self.std * torch.randn_like(self.mu)

    def mode(self) -> torch.Tensor:
        return self.mu

    def kl(
        self,
        other: Optional["DiagonalGaussianSeparate"] = None,
        reduce_dims: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        dims = tuple(reduce_dims) if reduce_dims is not None else tuple(range(1, self.mu.ndim))
        mu = self.mu.float()
        var = self.var.float()
        logvar = self.logvar.float()
        if other is None:
            out = 0.5 * torch.sum(mu.pow(2) + var - 1.0 - logvar, dim=dims)
        else:
            other_mu = other.mu.float()
            other_var = other.var.float()
            other_logvar = other.logvar.float()
            out = 0.5 * torch.sum(
                (mu - other_mu).pow(2) / other_var + var / other_var - 1.0 - logvar + other_logvar,
                dim=dims,
            )
        return out.to(dtype=self.mu.dtype)

    def nll(self, x: torch.Tensor, reduce_dims: Optional[Sequence[int]] = None) -> torch.Tensor:
        import math

        dims = tuple(reduce_dims) if reduce_dims is not None else tuple(range(1, self.mu.ndim))
        x_f = x.float()
        mu = self.mu.float()
        var = self.var.float()
        logvar = self.logvar.float()
        out = 0.5 * torch.sum(math.log(2.0 * math.pi) + logvar + (x_f - mu).pow(2) / var, dim=dims)
        return out.to(dtype=x.dtype)


@MODEL_REGISTRY.register("monai_vae")
class MonaiStyleVAE(BaseVAE):
    """MONAI-inspired KL VAE with separate mu/log-sigma projections."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        resolution: int = 256,
        channels: Sequence[int] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
        attention_levels: Sequence[bool] = (False, False, False, True),
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        spatial_dims: int = 2,
        input_range: str = "zero_to_one",
        scaling_factor: float = 1.0,
        disc_base_channels: int = 64,
        disc_n_layers: int = 3,
        attention_impl: str = "qkv",
        zero_init_attn_out: bool = True,
        block_factory=None,
    ) -> None:
        super().__init__()
        channels = tuple(int(ch) for ch in channels)
        attention_levels = tuple(bool(v) for v in attention_levels)
        if len(channels) != len(attention_levels):
            raise ValueError("channels and attention_levels must have the same length.")
        bad = [ch for ch in channels if ch % norm_num_groups != 0]
        if bad:
            raise ValueError(
                f"All channels must be divisible by norm_num_groups={norm_num_groups}. Offending channels: {bad}"
            )

        self.spatial_dims = spatial_dims
        self.input_range = str(input_range)
        self.scaling_factor = float(scaling_factor)
        self.latent_channels = int(latent_channels)
        self._disc_base_channels = int(disc_base_channels)
        self._disc_n_layers = int(disc_n_layers)

        attn_res = self._attention_levels_to_resolutions(attention_levels, resolution=int(resolution))
        ch_mult = self._channels_to_ch_mult(channels)

        self.encoder = Encoder(
            in_channels=in_channels,
            base_ch=channels[0],
            ch_mult=ch_mult,
            down_channels=channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res,
            resolution=resolution,
            z_channels=latent_channels,
            use_attention=any(attention_levels) or with_encoder_nonlocal_attn,
            spatial_dims=spatial_dims,
            norm_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_asymmetric_padding_downsample=True,
            double_z=True,
            zero_init_attn_out=zero_init_attn_out,
            attention_impl=attention_impl,
            block_factory=block_factory,
        )
        if not with_encoder_nonlocal_attn:
            self.encoder.mid_attn = nn.Identity()

        self.decoder = Decoder(
            out_ch=out_channels,
            base_ch=channels[0],
            ch_mult=ch_mult,
            down_channels=channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res,
            resolution=resolution,
            z_channels=latent_channels,
            use_attention=any(attention_levels) or with_decoder_nonlocal_attn,
            spatial_dims=spatial_dims,
            norm_groups=norm_num_groups,
            norm_eps=norm_eps,
            zero_init_attn_out=zero_init_attn_out,
            attention_impl=attention_impl,
            block_factory=block_factory,
        )
        if not with_decoder_nonlocal_attn:
            self.decoder.mid_attn = nn.Identity()

        self.quant_conv_mu = ConvND(spatial_dims, latent_channels, latent_channels, 1, padding=0)
        self.quant_conv_log_sigma = ConvND(spatial_dims, latent_channels, latent_channels, 1, padding=0)
        self.post_quant_conv = ConvND(spatial_dims, latent_channels, latent_channels, 1, padding=0)

    @staticmethod
    def _channels_to_ch_mult(channels: Sequence[int]) -> Tuple[int, ...]:
        base = int(channels[0])
        return tuple(int(ch // base) for ch in channels)

    @staticmethod
    def _attention_levels_to_resolutions(attention_levels: Sequence[bool], resolution: int = 256) -> Tuple[int, ...]:
        attn_res: list[int] = []
        curr_res = int(resolution)
        for idx, use_attn in enumerate(attention_levels):
            if use_attn:
                attn_res.append(curr_res)
            if idx < len(attention_levels) - 1:
                curr_res //= 2
        return tuple(attn_res)

    def make_discriminator(self) -> nn.Module:
        return SpectralNormDiscriminator(
            in_channels=self.decoder.conv_out.out_channels,
            base_channels=self._disc_base_channels,
            spatial_dims=self.spatial_dims,
            n_layers=self._disc_n_layers,
        )

    def encode(self, x: torch.Tensor, normalize: bool = False):
        h = self.encoder(x)
        h_mu, h_log_sigma = h.chunk(2, dim=1)
        z_mu = self.quant_conv_mu(h_mu)
        z_log_sigma = self.quant_conv_log_sigma(h_log_sigma).clamp(-30.0, 20.0)
        posterior = DiagonalGaussianSeparate(z_mu, torch.exp(z_log_sigma / 2.0), z_log_sigma)
        if normalize:
            return posterior.mode() * self.scaling_factor
        return posterior

    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if denorm:
            z = z / self.scaling_factor
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        posterior = self.encode(x, normalize=False)
        z = posterior.sample() if sample_posterior else posterior.mode()
        rec = self.decode(z, denorm=False)
        return ModelOutput(reconstruction=rec, posterior=posterior)
