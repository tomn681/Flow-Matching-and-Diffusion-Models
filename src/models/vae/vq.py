"""
Vector-quantized autoencoder assembled from modular VAE components.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from core.types import ModelOutput
from nn.modules.vae import Decoder, Encoder
from nn.modules.vae.codebook import VectorQuantizer, VectorQuantizerEMA
from nn.modules.vae.discriminators import MagvitDiscriminatorND
from nn.losses.vae import PatchDiscriminator
from nn.ops.convolution import ConvND
from ..registry import MODEL_REGISTRY
from .constants import LATENT_SCALE
from .base import BaseVAE


@MODEL_REGISTRY.register("vq_vae")
class VQVAE(BaseVAE):
    """
    Configurable VQ-VAE.

    Paper-level variants are expressed through config:
    - `quantizer_type`: `"classic"` for original VQ-VAE, `"ema"` for EMA-VQ/VQGAN-style tokenizers
    - `discriminator_type`: `"patchgan"` or `"magvit"`
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        resolution: int = 256,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        down_channels: Tuple[int, ...] | None = None,
        num_res_blocks: int = 2,
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
        norm_eps: float = 1e-6,
        zero_init_last_conv: bool = False,
        attention_impl: str = "compvis",
        input_range: str = "minus_one_to_one",
        scaling_factor: float = LATENT_SCALE,
        zero_init_attn_out: bool = True,
        latent_dropout: float = 0.0,
        use_asymmetric_padding_downsample: bool = True,
        codebook_size: int = 1024,
        vq_beta: float = 0.25,
        vq_ema_decay: float = 0.99,
        vq_ema_eps: float = 1e-5,
        quantizer_type: str = "ema",
        discriminator_type: str = "patchgan",
        block_factory=None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.input_range = str(input_range)
        self.scaling_factor = float(scaling_factor)
        self.latent_dropout = float(latent_dropout)
        self.quantizer_type = str(quantizer_type).lower()
        self.discriminator_type = str(discriminator_type).lower() if discriminator_type is not None else "patchgan"

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
            double_z=False,
            spatial_dims=spatial_dims,
            emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
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
            norm_eps=norm_eps,
            zero_init_last_conv=zero_init_last_conv,
            attention_impl=attention_impl,
            zero_init_attn_out=zero_init_attn_out,
            block_factory=block_factory,
        )

        self.quant_conv = ConvND(spatial_dims, z_channels, embed_dim, 1, padding=0)
        self.post_quant_conv = ConvND(spatial_dims, embed_dim, z_channels, 1, padding=0)
        self.embed_dim = embed_dim
        self.codebook = self._build_quantizer(
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            vq_beta=vq_beta,
            vq_ema_decay=vq_ema_decay,
            vq_ema_eps=vq_ema_eps,
        )

    def _build_quantizer(
        self,
        *,
        codebook_size: int,
        embed_dim: int,
        vq_beta: float,
        vq_ema_decay: float,
        vq_ema_eps: float,
    ):
        if self.quantizer_type in {"classic", "vq"}:
            return VectorQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=embed_dim,
                commitment_cost=vq_beta,
            )
        if self.quantizer_type == "ema":
            return VectorQuantizerEMA(
                num_embeddings=codebook_size,
                embedding_dim=embed_dim,
                commitment_cost=vq_beta,
                decay=vq_ema_decay,
                eps=vq_ema_eps,
            )
        raise ValueError(
            f"Unknown quantizer_type '{self.quantizer_type}'. Expected 'classic' or 'ema'."
        )

    def make_discriminator(self):
        """Select discriminator from config-backed model attributes."""
        if self.discriminator_type in {"patchgan", "default"}:
            return PatchDiscriminator(
                in_channels=self.decoder.conv_out.out_channels,
                spatial_dims=self.spatial_dims,
            )
        if self.discriminator_type == "magvit":
            return MagvitDiscriminatorND(
                in_channels=self.decoder.conv_out.out_channels,
                spatial_dims=self.spatial_dims,
            )
        raise ValueError(
            f"Unknown discriminator_type '{self.discriminator_type}'. Expected 'patchgan' or 'magvit'."
        )

    def encode(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        h = self.encoder(x)
        quant_in = self.quant_conv(h)
        if normalize:
            return quant_in * self.scaling_factor
        return quant_in

    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if denorm:
            z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> ModelOutput:
        # sample_posterior is accepted for API parity with KL-VAE and ignored for VQ-VAE.
        quant_in = self.encode(x, normalize=False)
        z_q, vq_loss, perplexity, codes = self.codebook(quant_in)
        if self.training and self.latent_dropout > 0.0:
            z_q = F.dropout2d(z_q, p=self.latent_dropout, training=True)
        rec = self.decode(z_q, denorm=False)
        return ModelOutput(
            reconstruction=rec,
            codebook_loss=vq_loss,
            auxiliary={"perplexity": perplexity, "codes": codes},
        )
