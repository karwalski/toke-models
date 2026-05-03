"""1B decoder-only transformer for toke code generation.

Architecture: 24 layers, 2048 hidden, 16 heads (GQA 4 KV), SwiGLU, RoPE, RMSNorm.

Primary framework: MLX (Apple Silicon native).
Fallback: PyTorch (for cloud GPU training).
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import mlx.core as mx
    import mlx.nn as nn

    _BACKEND = "mlx"
except ImportError:
    import torch
    import torch.nn as nn  # type: ignore[no-redef]

    _BACKEND = "torch"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TokeModelConfig:
    """Hyperparameters for the toke 1B model."""

    vocab_size: int = 8192
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4
    ffn_dim: int = 8192
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5


# ===================================================================
# MLX implementation
# ===================================================================

if _BACKEND == "mlx":

    class GroupQueryAttention(nn.Module):
        """Multi-head attention with grouped query attention (GQA).

        16 query heads share 4 key/value heads (4:1 ratio).
        """

        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.num_heads = cfg.num_heads
            self.num_kv_heads = cfg.num_kv_heads
            self.head_dim = cfg.hidden_dim // cfg.num_heads
            self.scale = self.head_dim ** -0.5

            self.q_proj = nn.Linear(cfg.hidden_dim, cfg.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_dim, bias=False)

            self.rope = nn.RoPE(self.head_dim, base=cfg.rope_base)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            B, L, _ = x.shape

            q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

            # Apply rotary position embeddings
            q = self.rope(q)
            k = self.rope(k)

            # Expand KV heads to match query heads (GQA)
            kv_repeat = self.num_heads // self.num_kv_heads
            if kv_repeat > 1:
                k = mx.repeat(k, kv_repeat, axis=1)
                v = mx.repeat(v, kv_repeat, axis=1)

            # Scaled dot-product attention
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None:
                attn = attn + mask
            attn = mx.softmax(attn, axis=-1)

            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(out)

    class SwiGLU(nn.Module):
        """SwiGLU feed-forward network.

        gate = SiLU(x @ W_gate)
        out  = (gate * (x @ W_up)) @ W_down
        """

        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.w_gate = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
            self.w_up = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
            self.w_down = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))

    class TransformerBlock(nn.Module):
        """Pre-norm transformer block: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual."""

        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.attn_norm = nn.RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.attn = GroupQueryAttention(cfg)
            self.ffn_norm = nn.RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.ffn = SwiGLU(cfg)

        def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
            x = x + self.attn(self.attn_norm(x), mask=mask)
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class TokeModel(nn.Module):
        """1B decoder-only transformer for toke code generation.

        Embedding -> 24 x TransformerBlock -> RMSNorm -> linear head.
        """

        def __init__(self, cfg: TokeModelConfig | None = None) -> None:
            super().__init__()
            if cfg is None:
                cfg = TokeModelConfig()
            self.cfg = cfg

            self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.layers = [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
            self.norm = nn.RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        def __call__(self, tokens: mx.array) -> mx.array:
            """Forward pass.

            Args:
                tokens: integer token IDs, shape (batch, seq_len).

            Returns:
                Logits, shape (batch, seq_len, vocab_size).
            """
            B, L = tokens.shape
            x = self.embedding(tokens)

            # Causal mask: upper-triangular -inf
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)

            for layer in self.layers:
                x = layer(x, mask=mask)

            x = self.norm(x)
            return self.head(x)

    def count_parameters(model: TokeModel) -> int:
        """Count total trainable parameters in the model."""
        leaves = model.parameters()
        return sum(p.size for p in _flatten(leaves))

    def _flatten(tree: dict | list | mx.array) -> list[mx.array]:
        """Recursively flatten a nested dict/list of mx.arrays."""
        if isinstance(tree, mx.array):
            return [tree]
        if isinstance(tree, dict):
            return [a for v in tree.values() for a in _flatten(v)]
        if isinstance(tree, list):
            return [a for v in tree for a in _flatten(v)]
        return []

# ===================================================================
# PyTorch fallback implementation
# ===================================================================

elif _BACKEND == "torch":
    import math

    class _RMSNorm(nn.Module):  # type: ignore[no-redef]
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))  # type: ignore[attr-defined]

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    def _precompute_freqs(dim: int, max_len: int, base: float = 10_000.0) -> torch.Tensor:  # type: ignore[name-defined]
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # type: ignore[attr-defined]
        t = torch.arange(max_len).float()  # type: ignore[attr-defined]
        freqs = torch.outer(t, freqs)  # type: ignore[attr-defined]
        return torch.polar(torch.ones_like(freqs), freqs)  # type: ignore[attr-defined]

    def _apply_rope(
        x: torch.Tensor, freqs: torch.Tensor  # type: ignore[name-defined]
    ) -> torch.Tensor:  # type: ignore[name-defined]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # type: ignore[attr-defined]
        freqs = freqs[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(x_complex * freqs).flatten(-2).type_as(x)  # type: ignore[attr-defined]

    class GroupQueryAttention(nn.Module):  # type: ignore[no-redef]
        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.num_heads = cfg.num_heads
            self.num_kv_heads = cfg.num_kv_heads
            self.head_dim = cfg.hidden_dim // cfg.num_heads
            self.scale = self.head_dim ** -0.5

            self.q_proj = nn.Linear(cfg.hidden_dim, cfg.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.hidden_dim, cfg.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_dim, bias=False)

            self.register_buffer(
                "freqs", _precompute_freqs(self.head_dim, cfg.max_seq_len, cfg.rope_base)
            )

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[name-defined]
            B, L, _ = x.shape
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = _apply_rope(q, self.freqs)
            k = _apply_rope(k, self.freqs)

            kv_repeat = self.num_heads // self.num_kv_heads
            if kv_repeat > 1:
                k = k.repeat_interleave(kv_repeat, dim=1)
                v = v.repeat_interleave(kv_repeat, dim=1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn + mask
            attn = torch.softmax(attn, dim=-1)  # type: ignore[attr-defined]

            out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
            return self.o_proj(out)

    class SwiGLU(nn.Module):  # type: ignore[no-redef]
        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.w_gate = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
            self.w_up = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
            self.w_down = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))  # type: ignore[attr-defined]

    class TransformerBlock(nn.Module):  # type: ignore[no-redef]
        def __init__(self, cfg: TokeModelConfig) -> None:
            super().__init__()
            self.attn_norm = _RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.attn = GroupQueryAttention(cfg)
            self.ffn_norm = _RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.ffn = SwiGLU(cfg)

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[name-defined]
            x = x + self.attn(self.attn_norm(x), mask=mask)
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class TokeModel(nn.Module):  # type: ignore[no-redef]
        def __init__(self, cfg: TokeModelConfig | None = None) -> None:
            super().__init__()
            if cfg is None:
                cfg = TokeModelConfig()
            self.cfg = cfg

            self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
            self.norm = _RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
            self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            B, L = tokens.shape
            x = self.embedding(tokens)

            mask = torch.full((L, L), float("-inf"), device=tokens.device)  # type: ignore[attr-defined]
            mask = torch.triu(mask, diagonal=1)  # type: ignore[attr-defined]

            for layer in self.layers:
                x = layer(x, mask=mask)

            x = self.norm(x)
            return self.head(x)

    def count_parameters(model: TokeModel) -> int:  # type: ignore[no-redef]
        return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TokeModelConfig()
    model = TokeModel(cfg)
    n_params = count_parameters(model)
    print(f"Backend: {_BACKEND}")
    print(f"Parameters: {n_params:,} ({n_params / 1e9:.3f}B)")

    # Forward pass with random input
    batch, seq_len = 2, 128
    if _BACKEND == "mlx":
        tokens = mx.random.randint(0, cfg.vocab_size, shape=(batch, seq_len))
        logits = model(tokens)
        mx.eval(logits)
        print(f"Input shape:  ({batch}, {seq_len})")
        print(f"Output shape: {logits.shape}")
        assert logits.shape == (batch, seq_len, cfg.vocab_size), f"Bad shape: {logits.shape}"
    else:
        tokens = torch.randint(0, cfg.vocab_size, (batch, seq_len))  # type: ignore[attr-defined]
        logits = model(tokens)
        print(f"Input shape:  ({batch}, {seq_len})")
        print(f"Output shape: {tuple(logits.shape)}")
        assert logits.shape == (batch, seq_len, cfg.vocab_size), f"Bad shape: {logits.shape}"

    print("Forward pass OK")
