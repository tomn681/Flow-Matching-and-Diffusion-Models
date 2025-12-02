"""
KL-regularised autoencoder assembled from modular VAE components.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from nn.modules.vae import Decoder, DiagonalGaussian, Encoder

LATENT_SCALE: float = 0.18215


class AutoencoderKL(nn.Module):
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
        double_z: bool = True,
        block_factory=None,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
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
            block_factory=block_factory,
        )
        self.decoder = Decoder(
            out_ch=out_channels,
            base_ch=base_ch,
            ch_mult=ch_mult,
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

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim

        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state)
        else:
            warnings.warn("[AutoencoderKL] No checkpoint provided. Random initialization.")

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

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x, normalize=False)
        z = posterior.sample() if sample_posterior else posterior.mode()
        rec = self.decode(z, denorm=False)
        return rec, posterior
