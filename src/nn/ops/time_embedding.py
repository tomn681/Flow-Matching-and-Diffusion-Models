import math
import torch

def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    *,
    flip_sin_to_cos: bool = True,
    freq_shift: int = 0,
):
    """
    Sinusoidal timestep embeddings.

    Attributes:
        - timesteps: [torch.Tensor] 1-D Tensor of N indices, one per batch element.
        - dim: [int] Output dimension.
        - max_period [int] Frequency parameter for embeddings generation.
        
    Returns:
        [torch.Tensor] Positional embeddings (N, dim)
    """
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / max(half - freq_shift, 1)
    args = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half:], embedding[:, :half]], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
