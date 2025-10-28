import math
import torch

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
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
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
