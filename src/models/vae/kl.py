"""
KL-regularised autoencoder assembled from modular VAE components.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from core.types import ModelOutput
from nn.modules.vae import Decoder, DiagonalGaussian, Encoder
from nn.losses.vae import PatchDiscriminator
from nn.ops.convolution import ConvND
from ..registry import MODEL_REGISTRY
from .constants import LATENT_SCALE
from .base import BaseVAE


@MODEL_REGISTRY.register("kl_vae")
class AutoencoderKL(BaseVAE):
    """
    Stable-Diffusion-style autoencoder with Gaussian latents.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        resolution: int = 256,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        down_channels: Tuple[int, ...] | None = None,
        num_res_blocks: int | Mapping[str, int] = 2,
        attn_resolutions: Tuple[int, ...] = (),
        z_channels: int = 4,
        embed_dim: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: int = 1,
        attn_dim_head: int | None = None,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        norm_groups: Optional[int] = None,
        norm_eps: float = 1e-6,
        zero_init_last_conv: bool = False,
        attention_impl: str = "compvis",
        input_range: str = "minus_one_to_one",
        zero_init_attn_out: bool = False,
        use_asymmetric_padding_downsample: bool = True,
        codebook_size: Optional[int] = None,
        num_embeddings: Optional[int] = None,
        double_z: bool = True,
        block_factory=None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_range = str(input_range)

        self.encoder = Encoder(
            in_channels=in_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
            down_channels=down_channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            z_channels=z_channels,
            dropout=dropout,
            use_attention=use_attention,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            double_z=double_z,
            spatial_dims=spatial_dims,
            emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
            zero_init_last_conv=zero_init_last_conv,
            attention_impl=attention_impl,
            zero_init_attn_out=zero_init_attn_out,
            use_asymmetric_padding_downsample=use_asymmetric_padding_downsample,
            block_factory=block_factory,
        )
        self.decoder = Decoder(
            out_ch=out_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
            down_channels=down_channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            z_channels=z_channels,
            dropout=dropout,
            use_attention=use_attention,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            tanh_out=False,
            spatial_dims=spatial_dims,
            emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            norm_groups=norm_groups,
            norm_eps=norm_eps,
            zero_init_last_conv=zero_init_last_conv,
            attention_impl=attention_impl,
            zero_init_attn_out=zero_init_attn_out,
            block_factory=block_factory,
        )

        self.quant_conv = ConvND(spatial_dims, 2 * z_channels, 2 * embed_dim, 1, padding=0)
        self.post_quant_conv = ConvND(spatial_dims, embed_dim, z_channels, 1, padding=0)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.codebook_size = codebook_size

    def make_discriminator(self):
        """Default PatchGAN-style discriminator."""
        return PatchDiscriminator(
            in_channels=self.decoder.conv_out.out_channels,
            spatial_dims=self.spatial_dims,
        )

    def encode(self, x: torch.Tensor, normalize: bool = False) -> Union[DiagonalGaussian, torch.Tensor]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussian(moments)
        if normalize:
            return posterior.mode() * LATENT_SCALE
        return posterior

    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if denorm:
            z = z / LATENT_SCALE
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        posterior = self.encode(x, normalize=False)
        z = posterior.sample() if sample_posterior else posterior.mode()
        rec = self.decode(z, denorm=False)
        return ModelOutput(reconstruction=rec, posterior=posterior)
