import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AvgPoolND
from .convolution import ConvND

class UpsampleND(nn.Module):
    """
    N-dimensional upsampling with optional convolution (uses ConvND).
    """

    def __init__(self, spatial_dims: int, channels: int, use_conv: bool = True):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        self.channels = channels
        self.use_conv = use_conv
        self.spatial_dims = spatial_dims

        if use_conv:
            self.conv = ConvND(spatial_dims, channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
        
class DownsampleND(nn.Module):
    """
    N-dimensional downsampling layer with optional convolution.

    Uses ConvND for learnable downsampling or AvgPoolND for fixed downsampling.
    """

    def __init__(self, spatial_dims: int, channels: int, use_conv: bool = True):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        self.channels = channels
        self.use_conv = use_conv
        self.spatial_dims = spatial_dims

        if use_conv:
            self.op = ConvND(
                spatial_dims,
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.op = AvgPoolND(spatial_dims, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


def run_self_tests() -> None:
    torch.manual_seed(0)

    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    shapes = {
        1: (2, 6, 33),
        2: (2, 6, 33, 35),
        3: (1, 4, 17, 21, 19),
    }

    for spatial_dims, shape in shapes.items():
        x = torch.randn(shape)
        channels = shape[1]

        up_no_conv = UpsampleND(spatial_dims, channels, use_conv=False)
        up_conv = UpsampleND(spatial_dims, channels, use_conv=True)

        y_no_conv = up_no_conv(x)
        y_conv = up_conv(x)
        expected_up = tuple(s * 2 for s in shape[2:])
        assert y_no_conv.shape == (shape[0], channels, *expected_up), "Upsample (no conv) shape mismatch"
        assert y_conv.shape == (shape[0], channels, *expected_up), "Upsample (conv) shape mismatch"

        down_conv = DownsampleND(spatial_dims, channels, use_conv=True)
        down_avg = DownsampleND(spatial_dims, channels, use_conv=False)

        z_conv = down_conv(x)
        z_avg = down_avg(x)
        expected_conv = tuple(_ceil_div(s, 2) for s in shape[2:])
        expected_avg = tuple(s // 2 for s in shape[2:])
        assert z_conv.shape == (shape[0], channels, *expected_conv), "Downsample (conv) shape mismatch"
        assert z_avg.shape == (shape[0], channels, *expected_avg), "Downsample (avg) shape mismatch"

        print(f"Upsample/Downsample variants passed for spatial_dims={spatial_dims}.")

    print("All upsampling module self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
