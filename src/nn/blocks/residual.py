import math
from typing import Optional

import torch
import torch.nn as nn

from nn.ops.convolution import ConvND
from .timestep import TimestepBlock


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockND(TimestepBlock):
    """
    Residual block with optional timestep/state conditioning.

    Attributes:
        channels -> [int] Input channels.
        emb_channels -> [int | None] Embedding channels when conditioning is used.
        dropout -> [float] Dropout rate.
        out_channels -> [int, default: None] Output channels. If None, uses channels.
        use_conv -> [bool, default: False] Whether to use a convolution in the skip branch.
        spatial_dims -> [int, default: 2] Dimensionality of the convolution (1/2/3D).
    """

    def __init__(
        self,
        channels: int,
        emb_channels: Optional[int],
        dropout: float,
        out_channels: int = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm and emb_channels is not None
        self.uses_embedding = emb_channels is not None

        if emb_channels is None and use_scale_shift_norm:
            raise ValueError("use_scale_shift_norm requires emb_channels to be provided.")

        norm_in_groups = max(1, math.gcd(channels, 32))
        norm_out_groups = max(1, math.gcd(self.out_channels, 32))

        self.in_layers = nn.Sequential(
            nn.GroupNorm(norm_in_groups, channels),
            nn.SiLU(),
            ConvND(spatial_dims, channels, self.out_channels, 3, padding=1),
        )

        if self.uses_embedding:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels if self.use_scale_shift_norm else self.out_channels,
                ),
            )
        else:
            self.emb_layers = None

        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_out_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                ConvND(spatial_dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = ConvND(spatial_dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = ConvND(spatial_dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply the block to a Tensor, optionally conditioned on an embedding.

        Args:
            x -> [torch.Tensor] Features (N, C, L) | (N, C, H, W) | (N, C, D, H, W)
            emb -> [torch.Tensor | None] Conditioning embeddings (N, emb_channels)

        Returns:
            [torch.Tensor] Outputs (N, out_channels, L) | (N, out_channels, H, W) | (N, out_channels, D, H, W)
        """
        h = self.in_layers(x)       # (N, out_channels, H, W)

        if self.uses_embedding:
            if emb is None:
                raise ValueError("ResBlockND expects `emb` when emb_channels is set.")
            emb_out = self.emb_layers(emb).type(h.dtype)
            # (N, 2*out_channels) if use_scale_shift_norm else (N, out_channels)
            emb_out = emb_out.view(*emb_out.shape, *([1] * (h.ndim - emb_out.ndim))) 
            # (N, 2*out_channels, 1, 1)  if use_scale_shift_norm else (N, out_channels, 1, 1)

            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1) # Both (N, out_channels, 1, 1)
                h = out_norm(h) * (1 + scale) + shift         # (N, out_channels, H, W)
                h = out_rest(h)                               # (N, out_channels, H, W)
            else:
                h = h + emb_out                               # (N, out_channels, H, W)
                h = self.out_layers(h)                        # (N, out_channels, H, W)
        else:
            h = self.out_layers(h)                            # (N, out_channels, H, W)

        return self.skip_connection(x) + h                    # (N, out_channels, H, W)


def run_self_tests() -> None:
    torch.manual_seed(0)

    spatial_map = {
        1: (1, 32, 33),
        2: (1, 32, 33, 33),
        3: (1, 16, 17, 17, 17),
    }
    emb = {
        1: torch.randn(1, 32),
        2: torch.randn(1, 32),
        3: torch.randn(1, 32),
    }

    configs = [
        dict(out_channels=None, use_conv=False, use_scale_shift_norm=False),
        dict(out_channels=None, use_conv=False, use_scale_shift_norm=True),
        dict(out_channels=64, use_conv=False, use_scale_shift_norm=False),
        dict(out_channels=64, use_conv=False, use_scale_shift_norm=True),
        dict(out_channels=64, use_conv=True, use_scale_shift_norm=False),
        dict(out_channels=64, use_conv=True, use_scale_shift_norm=True),
    ]

    for spatial_dims, shape in spatial_map.items():
        x = torch.randn(shape)
        t_emb = emb[spatial_dims]
        in_channels = shape[1]
        print(f"\n--- Testing spatial_dims={spatial_dims} ---")

        for cfg in configs:
            model = ResBlockND(
                spatial_dims=spatial_dims,
                channels=in_channels,
                emb_channels=t_emb.shape[1],
                dropout=0.1,
                **cfg,
            )
            out = model(x, t_emb)
            expected_c = cfg["out_channels"] or in_channels
            assert out.shape[1] == expected_c, f"Expected {expected_c}, got {out.shape[1]}"
            assert out.shape[2:] == x.shape[2:], "Spatial shape mismatch"

        plain = ResBlockND(
            spatial_dims=spatial_dims,
            channels=in_channels,
            emb_channels=None,
            dropout=0.1,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
        )
        out_plain = plain(x)
        assert out_plain.shape[1] == in_channels and out_plain.shape[2:] == x.shape[2:], "Unconditional block mismatch"
        print(f"ResBlockND variants passed for spatial_dims={spatial_dims}.")

    print("All ResBlockND self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
