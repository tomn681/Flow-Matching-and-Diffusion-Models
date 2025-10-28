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


def run_self_tests() -> None:
    torch.manual_seed(0)

    def _expected_down_shape(shape: Tuple[int, ...], factor) -> Tuple[int, ...]:
        spatial = shape[2:]
        if isinstance(factor, int):
            factors = (factor,) * len(spatial)
        else:
            factors = factor
        down_spatial = tuple(max(1, s // f) for s, f in zip(spatial, factors))
        return (shape[0], shape[1], *down_spatial)

    def _expected_up_shape(shape: Tuple[int, ...], factor) -> Tuple[int, ...]:
        spatial = shape[2:]
        if isinstance(factor, int):
            factors = (factor,) * len(spatial)
        else:
            factors = factor
        up_spatial = tuple(s * f for s, f in zip(spatial, factors))
        return (shape[0], shape[1], *up_spatial)

    shapes = {
        1: (2, 8, 17),
        2: (2, 8, 16, 18),
        3: (1, 4, 8, 10, 12),
    }
    factors = {
        1: 2,
        2: 2,
        3: (2, 1, 3),
    }

    for spatial_dims, shape in shapes.items():
        x = torch.randn(shape)
        c = shape[1]
        factor = factors[spatial_dims]

        # PoolND variants
        pool_identity = PoolND(spatial_dims, c, c, 1)
        assert torch.allclose(pool_identity(x), x), f"Pool identity failed for dims={spatial_dims}"

        pool = PoolND(spatial_dims, c, c, factor)
        y = pool(x)
        assert y.shape == _expected_down_shape(shape, factor), f"PoolND shape mismatch for dims={spatial_dims}"

        # Avg/Max pooling wrappers
        avg = AvgPoolND(spatial_dims, kernel_size=2)
        avg_out = avg(x)
        assert avg_out.shape[0] == shape[0] and avg_out.shape[1] == shape[1], "AvgPool batch/channel mismatch"

        max_pool = MaxPoolND(spatial_dims, kernel_size=2)
        max_out = max_pool(x)
        assert max_out.shape[0] == shape[0] and max_out.shape[1] == shape[1], "MaxPool batch/channel mismatch"

        # UnPoolND variants
        up_identity = UnPoolND(spatial_dims, c, c, 1)
        assert torch.allclose(up_identity(y), y), f"UnPool identity failed for dims={spatial_dims}"

        up = UnPoolND(spatial_dims, c, c, factor)
        z = up(y)
        assert z.shape == _expected_up_shape(y.shape, factor), f"UnPoolND shape mismatch for dims={spatial_dims}"

        print(f"Pooling variants passed for spatial_dims={spatial_dims}.")

    print("All pooling module self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
