import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AvgPoolND
from .convolution import ConvND

class UpsampleND(nn.Module):
    """
    N-dimensional upsampling with optional convolution (uses ConvND).
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        use_conv: bool = True,
        *,
        out_channels: int | None = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.spatial_dims = spatial_dims
        self.interpolate = bool(interpolate)

        if use_conv:
            self.conv = ConvND(
                spatial_dims,
                channels,
                self.out_channels,
                kernel_size=3,
                padding=padding,
                bias=bias,
            )

    def forward(self, x: torch.Tensor, output_size: int | tuple[int, ...] | None = None) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.interpolate:
            if output_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=output_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
        
class DownsampleND(nn.Module):
    """
    N-dimensional downsampling layer with optional convolution.

    Uses ConvND for learnable downsampling or AvgPoolND for fixed downsampling.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        use_conv: bool = True,
        *,
        out_channels: int | None = None,
        padding: int = 1,
        bias: bool = True,
        use_asymmetric_padding: bool = False,
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.spatial_dims = spatial_dims
        self.use_asymmetric_padding = bool(use_asymmetric_padding)

        if use_conv:
            padding = 0 if self.use_asymmetric_padding and spatial_dims == 2 else padding
            self.op = ConvND(
                spatial_dims,
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
                bias=bias,
            )
        else:
            if self.out_channels != channels:
                raise ValueError("DownsampleND without convolution requires out_channels == channels.")
            self.op = AvgPoolND(spatial_dims, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.use_conv and self.use_asymmetric_padding and self.spatial_dims == 2:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
        return self.op(x)
