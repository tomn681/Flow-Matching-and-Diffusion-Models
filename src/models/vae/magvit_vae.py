# my_vae.py — VAE con tus bloques N-D
# -----------------------------------------------------------------------------
# KL-Autoencoder compacto para imágenes, usando:
#  - ResBlockND (residual + GN + SiLU + dropout + skip conv)
#  - SpatialSelfAttention (MHSA espacial con SDPA de PyTorch)
#  - UpsampleND / DownsampleND (×2, N-D)
# Mantiene LATENT_SCALE y la API encode/decode/forward.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import math
import warnings
from typing import Iterable, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# === tus módulos ===
from nn.blocks.attention import SpatialSelfAttention
from nn.blocks.residual import ResBlockND
from nn.ops.upsampling import UpsampleND, DownsampleND
from nn.ops.convolution import ConvND
from .quantize import VectorQuantizerEMA

# -----------------------------------------------------------------------------#
# Constantes
# -----------------------------------------------------------------------------#

LATENT_SCALE: float = 0.18215  # convención SD para escalar latentes

# -----------------------------------------------------------------------------#
# Distribución Gaussiana diagonal (posterior)
# -----------------------------------------------------------------------------#

class DiagonalGaussian:
    """
    q(z|x) diagonal con helpers (sample, mode, KL, NLL).
    `parameters` = (B, 2*C, H, W) con [mu, logvar] en el eje de canales.
    """
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        mu, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        self.mu      = mu
        self.logvar  = logvar
        self.deter   = deterministic
        self.device  = parameters.device

        if deterministic:
            self.std = torch.zeros_like(mu, device=self.device)
            self.var = torch.zeros_like(mu, device=self.device)
        else:
            self.std = torch.exp(0.5 * logvar)
            self.var = torch.exp(logvar)

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        if self.deter:
            return self.mu
        return self.mu + self.std * torch.randn_like(self.mu, device=self.device)

    def mode(self) -> torch.Tensor:
        return self.mu

    def kl(self, other: Optional["DiagonalGaussian"]=None, reduce_dims: Iterable[int]=(1,2,3)) -> torch.Tensor:
        if self.deter:
            return torch.tensor([0.0], device=self.device)
        if other is None:
            return 0.5 * torch.sum(self.mu.pow(2) + self.var - 1.0 - self.logvar, dim=reduce_dims)
        return 0.5 * torch.sum(
            (self.mu - other.mu).pow(2) / other.var +
            self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=reduce_dims
        )

    def nll(self, x: torch.Tensor, reduce_dims: Iterable[int]=(1,2,3)) -> torch.Tensor:
        logtwopi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + (x - self.mu).pow(2) / self.var, dim=reduce_dims)

# -----------------------------------------------------------------------------#
# Encoder / Decoder con tus bloques
# -----------------------------------------------------------------------------#

class Encoder(nn.Module):
    """
    Encoder jerárquico con ResBlockND + (opcional) SpatialSelfAttention.
    Nota: ResBlockND ahora admite conditioning opcional; si `emb_channels` es None, se omite.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        resolution: int = 256,
        z_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: Optional[int] = None,
        attn_dim_head: Optional[int] = None,
        double_z: bool = True,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        norm_groups: Optional[int] = None,
    ):
        super().__init__()
        self.resolution   = resolution
        self.double_z     = double_z
        self.z_channels   = z_channels
        self.spatial_dims = spatial_dims
        self.emb_channels = emb_channels
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        if emb_channels is None and use_scale_shift_norm:
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")

        self.conv_in = ConvND(spatial_dims, in_channels, base_ch, 3, padding=1)

        curr_res = resolution
        in_ch = base_ch
        downs: List[nn.Module] = []
        for mult in ch_mult:
            out_ch = base_ch * mult
            blocks = []
            attns  = []
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlockND(
                        channels=in_ch,
                        emb_channels=emb_channels,
                        dropout=dropout,
                        out_channels=out_ch,
                        use_conv=False,
                        use_scale_shift_norm=use_scale_shift_norm,
                        spatial_dims=spatial_dims,
                    )
                )
                in_ch = out_ch
                if use_attention and (curr_res in attn_resolutions):
                    attns.append(self._build_attention_layer(in_ch))
            stage = nn.Module()
            stage.blocks = nn.ModuleList(blocks)
            stage.attns  = nn.ModuleList(attns)
            if mult != ch_mult[-1]:
                stage.down = DownsampleND(spatial_dims, in_ch, use_conv=True)
                curr_res //= 2
            downs.append(stage)
        self.downs = nn.ModuleList(downs)

        # middle
        self.mid_block1 = ResBlockND(
            channels=in_ch,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=in_ch,
            use_conv=False,
            use_scale_shift_norm=use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )
        self.mid_attn   = self._build_attention_layer(in_ch) if use_attention else nn.Identity()
        self.mid_block2 = ResBlockND(
            channels=in_ch,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=in_ch,
            use_conv=False,
            use_scale_shift_norm=use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )

        # out heads
        norm_groups = norm_groups or max(1, math.gcd(in_ch, 32))
        self.norm_out = nn.GroupNorm(norm_groups, in_ch)
        out_ch = 2 * z_channels if double_z else z_channels
        self.conv_out = ConvND(spatial_dims, in_ch, out_ch, 3, padding=1)

    def _build_attention_layer(self, channels: int) -> nn.Module:
        heads = self.attn_heads if self.attn_heads is not None else 1
        if self.attn_dim_head is not None:
            dim_head = self.attn_dim_head
        elif heads == 1:
            dim_head = channels
        else:
            dim_head = max(1, channels // heads)
        return SpatialSelfAttention(
            dim=channels,
            heads=heads,
            dim_head=dim_head,
            use_linear=False,
            use_efficient_attn=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: Optional[torch.Tensor]
        if self.emb_channels is None:
            emb = None
        else:
            emb = torch.zeros(x.size(0), self.emb_channels, dtype=x.dtype, device=x.device)

        h = self.conv_in(x)
        curr = h
        for stage in self.downs:
            for i, block in enumerate(stage.blocks):
                curr = block(curr, emb)
                if i < len(stage.attns):
                    curr = stage.attns[i](curr)
            if hasattr(stage, "down"):
                curr = stage.down(curr)

        h = self.mid_block1(curr, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    """
    Decoder simétrico con ResBlockND + (opcional) SpatialSelfAttention.
    """
    def __init__(
        self,
        out_ch: int = 3,
        base_ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (),
        resolution: int = 256,
        z_channels: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: Optional[int] = None,
        attn_dim_head: Optional[int] = None,
        tanh_out: bool = False,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.resolution   = resolution
        self.tanh_out     = tanh_out
        self.spatial_dims = spatial_dims
        self.emb_channels = emb_channels
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        if emb_channels is None and use_scale_shift_norm:
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")

        lowest_res = resolution // (2 ** (len(ch_mult) - 1))
        block_in = base_ch * ch_mult[-1]

        self.conv_in = ConvND(spatial_dims, z_channels, block_in, 3, padding=1)

        self.mid_block1 = ResBlockND(
            channels=block_in,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=block_in,
            use_conv=False,
            use_scale_shift_norm=use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )
        self.mid_attn   = self._build_attention_layer(block_in) if use_attention else nn.Identity()
        self.mid_block2 = ResBlockND(
            channels=block_in,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=block_in,
            use_conv=False,
            use_scale_shift_norm=use_scale_shift_norm,
            spatial_dims=spatial_dims,
        )

        ups: List[nn.Module] = []
        in_ch = block_in
        curr_res = lowest_res
        for mult in reversed(ch_mult):
            out_ch_stage = base_ch * mult
            blocks = []
            attns  = []
            for _ in range(num_res_blocks + 1):
                blocks.append(
                    ResBlockND(
                        channels=in_ch,
                        emb_channels=emb_channels,
                        dropout=dropout,
                        out_channels=out_ch_stage,
                        use_conv=False,
                        use_scale_shift_norm=use_scale_shift_norm,
                        spatial_dims=spatial_dims,
                    )
                )
                in_ch = out_ch_stage
                if use_attention and (curr_res in attn_resolutions):
                    attns.append(self._build_attention_layer(in_ch))
            stage = nn.Module()
            stage.blocks = nn.ModuleList(blocks)
            stage.attns  = nn.ModuleList(attns)
            if mult != ch_mult[0]:
                stage.up = UpsampleND(spatial_dims, in_ch, use_conv=True)
                curr_res *= 2
            ups.insert(0, stage)  # prepend para mantener orden natural
        self.ups = nn.ModuleList(ups)

        norm_groups = max(1, math.gcd(in_ch, 32))
        self.norm_out = nn.GroupNorm(norm_groups, in_ch)
        self.conv_out = ConvND(spatial_dims, in_ch, out_ch, 3, padding=1)

    def _build_attention_layer(self, channels: int) -> nn.Module:
        heads = self.attn_heads if self.attn_heads is not None else 1
        if self.attn_dim_head is not None:
            dim_head = self.attn_dim_head
        elif heads == 1:
            dim_head = channels
        else:
            dim_head = max(1, channels // heads)
        return SpatialSelfAttention(
            dim=channels,
            heads=heads,
            dim_head=dim_head,
            use_linear=False,
            use_efficient_attn=True,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        emb: Optional[torch.Tensor]
        if self.emb_channels is None:
            emb = None
        else:
            emb = torch.zeros(z.size(0), self.emb_channels, dtype=z.dtype, device=z.device)

        h = self.conv_in(z)
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        curr = h
        for stage in reversed(self.ups):
            for i, block in enumerate(stage.blocks):
                curr = block(curr, emb)
                if i < len(stage.attns):
                    curr = stage.attns[i](curr)
            if hasattr(stage, "up"):
                curr = stage.up(curr)

        h = F.silu(self.norm_out(curr))
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h

# -----------------------------------------------------------------------------#
# KL-regularized Autoencoder
# -----------------------------------------------------------------------------#

class AutoencoderKL(nn.Module):
    """
    KL-regularized autoencoder con escalado SD del latente.
    - encode(x, normalize=False) -> DiagonalGaussian (o z*LATENT_SCALE si normalize=True)
    - decode(z, denorm=False)    -> reconstrucción desde latente
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
        embed_dim: int = 4,          # canales del latente “cuantizado”
        dropout: float = 0.0,
        use_attention: bool = True,
        attn_heads: int = 4,
        attn_dim_head: int = 64,
        spatial_dims: int = 2,
        emb_channels: Optional[int] = None,
        use_scale_shift_norm: bool = False,
        ckpt_path: Optional[str] = None,
        double_z: bool = True,
        latent_type: str = "kl",      # "kl" o "vq"
        codebook_size: Optional[int] = None,
        vq_beta: float = 0.25,
        vq_ema_decay: float = 0.99,
        vq_ema_eps: float = 1e-5,
    ):
        super().__init__()
        latent_type = latent_type.lower()
        if latent_type not in ("kl", "vq"):
            raise ValueError(f"latent_type must be 'kl' or 'vq', got {latent_type}")
        self.latent_type = latent_type

        if self.latent_type == "vq":
            double_z = False  # no mu/logvar heads for VQ

        if self.latent_type == "vq" and double_z:
            warnings.warn("[AutoencoderKL] latent_type='vq' forces double_z=False for codebook latents.")
            double_z = False

        self.encoder = Encoder(
            in_channels=in_channels, base_ch=base_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            resolution=resolution, z_channels=z_channels, dropout=dropout,
            use_attention=use_attention, attn_heads=attn_heads, attn_dim_head=attn_dim_head,
            double_z=double_z, spatial_dims=spatial_dims, emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.decoder = Decoder(
            out_ch=out_channels, base_ch=base_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            resolution=resolution, z_channels=z_channels, dropout=dropout,
            use_attention=use_attention, attn_heads=attn_heads, attn_dim_head=attn_dim_head,
            tanh_out=False, spatial_dims=spatial_dims, emb_channels=emb_channels,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        # Cabezas estilo SD (quant / post-quant)
        quant_in = z_channels if self.latent_type == "vq" else 2 * z_channels
        quant_out = embed_dim if self.latent_type == "vq" else 2 * embed_dim
        self.quant_conv      = nn.Conv2d(quant_in, quant_out, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.codebook: Optional[VectorQuantizerEMA]
        if self.latent_type == "vq":
            if codebook_size is None or codebook_size <= 0:
                raise ValueError("codebook_size must be set to a positive integer for VQ latent_type.")
            self.codebook = VectorQuantizerEMA(
                num_embeddings=codebook_size,
                embedding_dim=embed_dim,
                commitment_cost=vq_beta,
                decay=vq_ema_decay,
                eps=vq_ema_eps,
            )
        else:
            self.codebook = None

        if ckpt_path:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state)
        else:
            warnings.warn("[AutoencoderKL] No checkpoint provided. Random initialization.")

    # ---- API ----

    def encode(self, x: torch.Tensor, normalize: bool=False) -> Union[DiagonalGaussian, torch.Tensor]:
        """
        x: (B, C, H, W) en [-1, 1].
        - KL modo: devuelve DiagonalGaussian (o z_mode * LATENT_SCALE si normalize=True).
        - VQ modo: devuelve latentes cuantizados (opcionalmente escalados por LATENT_SCALE).
        """
        h = self.encoder(x)
        quant = self.quant_conv(h)
        if self.latent_type == "vq":
            if self.codebook is None:
                raise RuntimeError("VQ latent_type selected but codebook is missing.")
            z_q, _, _, _ = self.codebook(quant)
            return z_q * LATENT_SCALE if normalize else z_q

        posterior = DiagonalGaussian(quant)
        if normalize:
            return posterior.mode() * LATENT_SCALE
        return posterior

    def decode(self, z: torch.Tensor, denorm: bool=False) -> torch.Tensor:
        """
        z: (B, embed_dim, H/8, W/8). Si `denorm=True`, divide por LATENT_SCALE antes de decodificar.
        """
        if denorm:
            z = z / LATENT_SCALE
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool=True):
        if self.latent_type == "vq":
            if self.codebook is None:
                raise RuntimeError("VQ latent_type selected but codebook is missing.")
            h = self.encoder(x)
            quant = self.quant_conv(h)
            z_q, vq_loss, perplexity, codes = self.codebook(quant)
            rec = self.decode(z_q, denorm=False)
            vq_info = {
                "loss": vq_loss,
                "perplexity": perplexity,
                "codes": codes,
            }
            return rec, None, vq_info

        posterior = self.encode(x, normalize=False)
        z = posterior.sample() if sample_posterior else posterior.mode()
        rec = self.decode(z, denorm=False)
        return rec, posterior, None

# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL(
        in_channels=3, out_channels=3, resolution=256,
        base_ch=128, ch_mult=(1,2,4,4), num_res_blocks=2,
        attn_resolutions=(), z_channels=4, embed_dim=4,
        use_attention=True, spatial_dims=2, emb_channels=128,
        ckpt_path=None
    ).to(dev)

    x = torch.randn(1, 3, 256, 256, device=dev)
    rec, post = model(x)
    z = post.sample()
    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input    : {x.shape}")
    print(f"Latent   : {z.shape}")
    print(f"Output   : {rec.shape}")
