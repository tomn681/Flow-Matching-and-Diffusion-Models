"""
Vector quantization codebook modules for VQ-VAEs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class _VectorQuantizerBase(nn.Module):
    """Shared utilities for direct-gradient and EMA vector quantizers."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        *,
        l2_normalize_codes: bool = False,
        dead_code_threshold: float = 0.0,
        revive_dead_codes: bool = False,
        track_usage: bool = True,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.l2_normalize_codes = bool(l2_normalize_codes)
        self.dead_code_threshold = float(dead_code_threshold)
        self.revive_dead_codes = bool(revive_dead_codes)
        self.track_usage = bool(track_usage)
        self.last_telemetry: dict[str, torch.Tensor | float] = {}

    def _initial_embedding(self) -> torch.Tensor:
        bound = 1.0 / max(1, self.num_embeddings)
        emb = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(emb, -bound, bound)
        if self.l2_normalize_codes:
            emb = F.normalize(emb, dim=1)
        return emb

    def _distance_inputs(self, flat_z: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.l2_normalize_codes:
            return flat_z, embedding
        return F.normalize(flat_z, dim=1), F.normalize(embedding, dim=1)

    def _flatten(self, z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        permute_order = list(range(z.ndim))
        channel_dim = permute_order.pop(1)
        permute_order.append(channel_dim)
        z_perm = z.permute(*permute_order).contiguous()
        flat = z_perm.view(-1, z_perm.shape[-1])
        inverse = [0] * len(permute_order)
        for i, p in enumerate(permute_order):
            inverse[p] = i
        return flat, tuple(permute_order), tuple(inverse)

    def _restore(
        self,
        quantized_flat: torch.Tensor,
        z: torch.Tensor,
        permute_order: tuple[int, ...],
        inverse_permute: tuple[int, ...],
    ) -> torch.Tensor:
        quantized = quantized_flat.view(z.permute(*permute_order).shape)
        quantized = quantized.permute(*inverse_permute).contiguous()
        return z + (quantized - z).detach()

    def _stats(
        self,
        encodings: torch.Tensor,
        encoding_indices: torch.Tensor,
        z: torch.Tensor,
        permute_order: tuple[int, ...],
        inverse_permute: tuple[int, ...],
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del inverse_permute
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))
        codes = encoding_indices.view(z.permute(*permute_order).shape[:-1])
        if self.track_usage:
            used = torch.count_nonzero(avg_probs > 0).to(dtype=torch.float32)
            usage_fraction = used / float(self.num_embeddings)
            dead_fraction = 1.0 - usage_fraction
            self.last_telemetry = {
                "usage_fraction": float(usage_fraction.item()),
                "dead_code_fraction": float(dead_fraction.item()),
                "perplexity": float(perplexity.item()),
            }
        return perplexity, codes

    def _revive_dead_codes(self, flat_z: torch.Tensor, inactive_mask: torch.Tensor) -> None:
        if not self.revive_dead_codes or not bool(torch.any(inactive_mask)):
            return
        candidates = flat_z.detach()
        if candidates.numel() == 0:
            return
        num_dead = int(inactive_mask.sum().item())
        choice = torch.randint(0, candidates.size(0), (num_dead,), device=candidates.device)
        replacement = candidates[choice]
        if self.l2_normalize_codes:
            replacement = F.normalize(replacement, dim=1)
        self.embedding[inactive_mask] = replacement.to(self.embedding.dtype)


class VectorQuantizer(_VectorQuantizerBase):
    """
    Original VQ-VAE quantizer with direct codebook gradients.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        *,
        l2_normalize_codes: bool = False,
        dead_code_threshold: float = 0.0,
        revive_dead_codes: bool = False,
        track_usage: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            l2_normalize_codes=l2_normalize_codes,
            dead_code_threshold=dead_code_threshold,
            revive_dead_codes=revive_dead_codes,
            track_usage=track_usage,
        )
        self.embedding = nn.Parameter(self._initial_embedding())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_z, permute_order, inverse_permute = self._flatten(z)
        flat_distance_z, distance_embedding = self._distance_inputs(flat_z, self.embedding)

        z_sq = torch.sum(flat_distance_z ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(distance_embedding ** 2, dim=1)
        distances = z_sq + e_sq - 2.0 * torch.matmul(flat_distance_z, distance_embedding.t())

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)
        quantized_flat = torch.matmul(encodings, self.embedding)
        quantized = self._restore(quantized_flat, z, permute_order, inverse_permute)

        commitment_loss = F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        perplexity, codes = self._stats(encodings, encoding_indices, z, permute_order, inverse_permute)
        if self.revive_dead_codes and self.dead_code_threshold > 0.0:
            usage = encodings.sum(dim=0)
            inactive_mask = usage <= self.dead_code_threshold
            with torch.no_grad():
                self._revive_dead_codes(flat_z, inactive_mask)
        return quantized, vq_loss, perplexity, codes


class VectorQuantizerEMA(_VectorQuantizerBase):
    """
    Codebook quantizer with EMA updates.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        *,
        l2_normalize_codes: bool = False,
        dead_code_threshold: float = 0.0,
        revive_dead_codes: bool = False,
        track_usage: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            l2_normalize_codes=l2_normalize_codes,
            dead_code_threshold=dead_code_threshold,
            revive_dead_codes=revive_dead_codes,
            track_usage=track_usage,
        )
        self.decay = decay
        self.eps = eps

        embedding = self._initial_embedding()
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_z, permute_order, inverse_permute = self._flatten(z)

        flat_distance_z, distance_embedding = self._distance_inputs(flat_z, self.embedding)
        z_sq = torch.sum(flat_distance_z ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(distance_embedding ** 2, dim=1)
        distances = z_sq + e_sq - 2.0 * torch.matmul(flat_distance_z, distance_embedding.t())

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)
        quantized_flat = torch.matmul(encodings, self.embedding)

        if self.training and self.decay > 0.0:
            encodings_sum = torch.sum(encodings, dim=0)
            dw = torch.matmul(encodings.t(), flat_z)
            if utils.is_distributed():
                encodings_sum = utils.all_reduce_tensor(encodings_sum)
                dw = utils.all_reduce_tensor(dw)

            self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = torch.sum(self.ema_cluster_size)
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            updated_embedding = self.ema_w / cluster_size.unsqueeze(1)
            if self.l2_normalize_codes:
                updated_embedding = F.normalize(updated_embedding, dim=1)
            self.embedding.copy_(updated_embedding)
            if self.revive_dead_codes and self.dead_code_threshold > 0.0:
                inactive_mask = self.ema_cluster_size <= self.dead_code_threshold
                self._revive_dead_codes(flat_z, inactive_mask)
                self.ema_w[inactive_mask] = self.embedding[inactive_mask]
                self.ema_cluster_size[inactive_mask] = max(self.dead_code_threshold, 1.0)

        quantized = self._restore(quantized_flat, z, permute_order, inverse_permute)

        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = self.commitment_cost * commitment_loss

        perplexity, codes = self._stats(encodings, encoding_indices, z, permute_order, inverse_permute, eps=self.eps)
        return quantized, vq_loss, perplexity, codes
