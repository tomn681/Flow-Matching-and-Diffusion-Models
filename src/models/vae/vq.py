"""
Vector-quantized autoencoder assembled from modular VAE components.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch

from nn.modules.vae import Decoder, Encoder
from nn.modules.vae.codebook import VectorQuantizer, VectorQuantizerEMA
from nn.modules.vae.discriminators import MagvitDiscriminatorND
from nn.losses.vae import PatchDiscriminator
from nn.ops.convolution import ConvND
from .base import BaseVAE

LATENT_SCALE: float = 0.18215


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
        attn_heads: int = 4,
        attn_dim_head: int = 64,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        ckpt_path: Optional[str] = None,
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

        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state)
        else:
            warnings.warn("[VQVAE] No checkpoint provided. Random initialization.")

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
            return quant_in * LATENT_SCALE
        return quant_in

    def decode(self, z: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if denorm:
            z = z / LATENT_SCALE
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        quant_in = self.encode(x, normalize=False)
        z_q, vq_loss, perplexity, codes = self.codebook(quant_in)
        rec = self.decode(z_q, denorm=False)
        return rec, {"vq_loss": vq_loss, "perplexity": perplexity, "codes": codes}
