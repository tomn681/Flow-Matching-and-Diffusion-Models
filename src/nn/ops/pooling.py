from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .convolution import ConvND, ConvTransposeND

SizeArg = Union[int, Tuple[int, ...]]

class PoolND(nn.Module):
    """
    Downsampling by ConvND with kernel=stride=pool_factor and padding=0.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, pool_factor: SizeArg = 2):
        super().__init__()
        if pool_factor == 1 or (isinstance(pool_factor, (tuple, list)) and all(p == 1 for p in pool_factor)):
            self.down = nn.Identity()
        else:
            self.down = ConvND(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=pool_factor,
                stride=pool_factor,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)
        
class AvgPoolND(nn.Module):
    """
    Envelope class for n-dimensional average pooling.
    """

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: SizeArg = 2,
        stride: Optional[SizeArg] = None,
        padding: SizeArg = 0,
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        pool_map = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
        Pool = pool_map[spatial_dims]
        self.pool = Pool(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
        
class MaxPoolND(nn.Module):
    """
    Envelope class for n-dimensional max pooling.
    """

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: SizeArg = 2,
        stride: Optional[SizeArg] = None,
        padding: SizeArg = 0,
        dilation: SizeArg = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        pool_map = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
        Pool = pool_map[spatial_dims]
        self.pool = Pool(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class UnPoolND(nn.Module):
    """
    Upsampling by ConvTransposeND with kernel=stride=pool_factor and padding=0.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, pool_factor: SizeArg = 2):
        super().__init__()
        if pool_factor == 1 or (isinstance(pool_factor, (tuple, list)) and all(p == 1 for p in pool_factor)):
            self.up = nn.Identity()
        else:
            self.up = ConvTransposeND(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=pool_factor,
                stride=pool_factor,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
