"""
Vector quantization modules for VQ-VAE / VQGAN.

Includes an EMA-updated codebook variant that mirrors common VQGAN practice:
 - Straight-through estimator for gradients.
 - Optional EMA updates for embedding vectors (set decay=0 to disable EMA).
 - Returns quantized latents, VQ loss, and perplexity stats.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    Codebook quantizer with optional EMA updates.

    Args:
        num_embeddings: Size of the codebook.
        embedding_dim: Dimension of each embedding vector.
        commitment_cost: Weight for the commitment loss term.
        decay: EMA decay; set to 0.0 to disable EMA and use direct codebook gradients.
        eps: Numerical stability epsilon.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def _flatten(self, z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
        """
        Move channels to the last dimension and flatten spatial dims.

        Returns flattened tensor, permutation order, and inverse permutation.
        """
        permute_order = list(range(z.ndim))
        channel_dim = permute_order.pop(1)
        permute_order.append(channel_dim)  # channels last
        z_perm = z.permute(*permute_order).contiguous()

        flat = z_perm.view(-1, z_perm.shape[-1])

        # Build inverse permutation
        inverse_permute = [0] * len(permute_order)
        for i, p in enumerate(permute_order):
            inverse_permute[p] = i

        return flat, tuple(permute_order), tuple(inverse_permute)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input `z` and return (quantized, vq_loss, perplexity, codes).

        `codes` is a tensor of shape (B, *spatial) with code indices.
        """
        flat_z, permute_order, inverse_permute = self._flatten(z)

        # Compute L2 distance to embeddings: ||z||^2 + ||e||^2 - 2 z·e
        z_sq = torch.sum(flat_z ** 2, dim=1, keepdim=True)  # (N, 1)
        e_sq = torch.sum(self.embedding ** 2, dim=1)  # (M,)
        distances = z_sq + e_sq - 2.0 * torch.matmul(flat_z, self.embedding.t())  # (N, M)

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)

        quantized_flat = torch.matmul(encodings, self.embedding)  # (N, D)

        if self.training and self.decay > 0.0:
            # EMA updates
            encodings_sum = torch.sum(encodings, dim=0)
            dw = torch.matmul(encodings.t(), flat_z)

            self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = torch.sum(self.ema_cluster_size)
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            self.embedding.copy_(self.ema_w / cluster_size.unsqueeze(1))

        elif self.training and self.decay == 0.0:
            # Direct codebook gradient (non-EMA)
            pass  # gradients flow through self.embedding via quantized_flat

        # Straight-through estimator
        quantized = quantized_flat.view(z.permute(*permute_order).shape)
        quantized = quantized.permute(*inverse_permute).contiguous()
        quantized = z + (quantized - z).detach()

        # Loss and stats
        commitment_loss = F.mse_loss(quantized.detach(), z)
        if self.decay == 0.0:
            codebook_loss = F.mse_loss(quantized, z.detach())
            vq_loss = commitment_loss * self.commitment_cost + codebook_loss
        else:
            vq_loss = commitment_loss * self.commitment_cost

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        codes = encoding_indices.view(z.permute(*permute_order).shape[:-1])
        codes = codes.permute(*inverse_permute)

        return quantized, vq_loss, perplexity, codes


__all__ = ["VectorQuantizerEMA"]
