from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

SizeArg = Union[int, Tuple[int, ...]]

class ConvND(nn.Module):
    """
    Envelope class for n-dimensional convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeArg = 3,
        stride: SizeArg = 1,
        padding: Optional[SizeArg] = None,
        dilation: SizeArg = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")
            
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)

        conv_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        try:
            conv = conv_map[spatial_dims]
        except KeyError:
            raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ConvTransposeND(nn.Module):
    """
    Envelope class for n-dimensional transposed convolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeArg = 2,
        stride: SizeArg = 2,
        padding: SizeArg = 0,
        output_padding: Optional[SizeArg] = None,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if spatial_dims not in (1, 2, 3):
            raise ValueError("spatial_dims must be 1, 2 or 3")

        convt_map = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
        try:
            convt = convt_map[spatial_dims]
        except KeyError:
            raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

        self.convT = convt(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding or 0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convT(x)
