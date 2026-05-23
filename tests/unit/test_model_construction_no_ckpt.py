import torch
import torch.nn as nn

from models.vae.kl import AutoencoderKL
from models.vae.vq import VQVAE


def test_kl_constructs_without_ckpt_param() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=8,
        base_ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        use_attention=False,
        z_channels=4,
        embed_dim=4,
        spatial_dims=2,
    )
    x = torch.randn(1, 1, 8, 8)
    out = model(x, sample_posterior=False)
    assert out.reconstruction.shape == x.shape


def test_vq_constructs_without_ckpt_param() -> None:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        resolution=8,
        base_ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        use_attention=False,
        z_channels=4,
        embed_dim=4,
        codebook_size=16,
        quantizer_type="classic",
        spatial_dims=2,
    )
    class _DummyCodebook(nn.Module):
        def forward(self, quant_in):
            return (
                quant_in,
                torch.tensor(0.0, device=quant_in.device),
                torch.tensor(1.0, device=quant_in.device),
                torch.zeros_like(quant_in[:, :1, ...], dtype=torch.long),
            )

    model.codebook = _DummyCodebook()
    x = torch.randn(1, 1, 8, 8)
    out = model(x)
    assert out.reconstruction.shape == x.shape
