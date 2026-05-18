import torch
import torch.nn as nn

from core.types import ModelOutput
from models.vae.kl import AutoencoderKL
from models.vae.vq import VQVAE


def test_kl_forward_returns_model_output() -> None:
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        resolution=8,
        base_ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        ckpt_path=None,
    )
    model.eval()
    x = torch.randn(2, 1, 8, 8)

    output = model(x, sample_posterior=False)

    assert isinstance(output, ModelOutput)
    assert output.reconstruction.shape == x.shape
    assert output.posterior is not None
    assert output.codebook_loss is None


def test_vq_forward_returns_model_output() -> None:
    model = VQVAE(
        in_channels=1,
        out_channels=1,
        resolution=8,
        base_ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        z_channels=4,
        embed_dim=4,
        use_attention=False,
        spatial_dims=2,
        codebook_size=16,
        quantizer_type="classic",
        ckpt_path=None,
    )
    model.eval()
    # Keep this test focused on ModelOutput contract, not codebook internals.
    class _DummyCodebook(nn.Module):
        def forward(self, quant_in):
            return (
                quant_in,
                torch.tensor(0.25, device=quant_in.device),
                torch.tensor(0.9, device=quant_in.device),
                torch.zeros_like(quant_in[:, :1, ...], dtype=torch.long),
            )

    model.codebook = _DummyCodebook()
    x = torch.randn(2, 1, 8, 8)

    output = model(x)

    assert isinstance(output, ModelOutput)
    assert output.reconstruction.shape == x.shape
    assert output.posterior is None
    assert output.codebook_loss is not None
    assert "perplexity" in output.auxiliary
    assert "codes" in output.auxiliary
