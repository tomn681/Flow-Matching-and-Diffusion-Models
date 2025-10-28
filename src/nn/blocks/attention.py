import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class QKVAttention(nn.Module):
    """
    Implementation of Scaled Dot Product Attention.
    
    Attributes:
        - efficient_attn [bool, default: True] Uses the efficient PyTorch implementation.
        - dropout [float, default: 0.0] Dropout value.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        has_sdp = hasattr(F, "scaled_dot_product_attention")
        self.efficient_attn = bool(efficient_attn and has_sdp)
        if efficient_attn and not has_sdp:
            warnings.warn(
                "Efficient scaled dot-product attention requires PyTorch >= 2.0. "
                "Falling back to the explicit implementation.",
                RuntimeWarning,
                stacklevel=2,
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q -> [torch.Tensor] Queries (N, ..., Q_heads, Tgt_dim, QK_emb)
            k -> [torch.Tensor] Keys (N, ..., KV_heads, S_dim, QK_emb)
            v -> [torch.Tensor] Values (N, ..., KV_heads, S_dim, V_emb)
            S_dim stands for source dimension.
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=False
            )

        scale = 1 / math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        return torch.matmul(attn, v)


class LinearQKVAttention(nn.Module):
    """
    Linear attention variant that factors the softmax to reduce memory usage.
    """

    def __init__(self, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.dropout = dropout
        self.eps = eps

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_softmax = F.softmax(k, dim=-2)
        q_softmax = F.softmax(q, dim=-1)

        context = torch.einsum("...nd,...ne->...de", k_softmax, v)
        context = context / (k_softmax.sum(dim=-2, keepdim=False).unsqueeze(-1) + self.eps)
        out = torch.einsum("...nd,...de->...ne", q_softmax, context)
        return F.dropout(out, p=self.dropout, training=self.training)


def zero_module(module):
    """
    Zero out the parameters of a module.
    
    Ported from CompVis fm-boosting as an extension of SpatialSelfAttention.
    https://github.com/CompVis/fm-boosting/tree/main
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialSelfAttention(nn.Module):
    """
    Implementation of Multi-head Spatial Self-Attention Block.
    
    Ported from CompVis fm-boosting
    https://github.com/CompVis/fm-boosting/tree/main
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64,
                 use_linear: bool = False, use_efficient_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(max(1, math.gcd(dim, 32)), dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.dim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (b, c, *spatial), where spatial can be (f, h, w) or (h, w).
        Returns:
            x: Tensor after attention, MHSA(x) + residual.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)                                     # (b, c, f * h * w)
        qkv = self.qkv(self.norm(x))                                # (b, 3 * c * nh, f * h * w)
        qkv = qkv.reshape(b, self.heads, qkv.shape[-1], -1)         # (b, nh, f * h * w, 3 * c)
        q, k, v = qkv.chunk(3, dim=-1)                              # (b, nh, f * h * w, c) each
        h = self.attention(q, k, v)                                 # (b, nh, f * h * w, c)
        h = h.reshape(b, self.inner_dim, -1)                        # (b, nh * c, f * h * w)
        h = self.proj_out(h)                                        # (b, c, f * h * w)
        return (x + h).reshape(b, c, *spatial)


def run_self_tests() -> None:
    torch.manual_seed(0)

    def _test_qkv_attention():
        batch, heads, tokens, dim = 2, 4, 8, 16
        q = torch.randn(batch, heads, tokens, dim)
        k = torch.randn(batch, heads, tokens, dim)
        v = torch.randn(batch, heads, tokens, dim)

        vanilla = QKVAttention(efficient_attn=False, dropout=0.0)
        vanilla_out = vanilla(q, k, v)
        assert vanilla_out.shape == q.shape, "Vanilla attention shape mismatch"

        efficient = QKVAttention(efficient_attn=True, dropout=0.0)
        efficient.eval()
        efficient_out = efficient(q, k, v)
        assert efficient_out.shape == q.shape, "Efficient attention shape mismatch"
        print("QKVAttention variants passed.")

        linear = LinearQKVAttention(dropout=0.0)
        linear_out = linear(q, k, v)
        assert linear_out.shape == q.shape, "Linear attention shape mismatch"
        print("LinearQKVAttention passed.")

    def _test_spatial_self_attention():
        configs = [
            dict(channels=32, spatial=(8, 8), use_linear=False),
            dict(channels=32, spatial=(8, 8), use_linear=True),
            dict(channels=16, spatial=(4, 8, 8), use_linear=False),
            dict(channels=16, spatial=(4, 8, 8), use_linear=True),
        ]

        for cfg in configs:
            shape = (1, cfg["channels"], *cfg["spatial"])
            x = torch.randn(shape)
            block = SpatialSelfAttention(
                dim=cfg["channels"],
                heads=4,
                dim_head=cfg["channels"] // 4,
                use_linear=cfg["use_linear"],
                use_efficient_attn=True,
            )
            y = block(x)
            assert y.shape == x.shape, f"SpatialSelfAttention failed for {cfg}"
        print("SpatialSelfAttention variants passed.")

    _test_qkv_attention()
    _test_spatial_self_attention()
    print("All attention module self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
