"""Tests for the toke 1B model definition."""

from __future__ import annotations

import sys
sys.path.insert(0, "/Users/matthew.watt/tk/toke-model")

from model.model import TokeModel, TokeModelConfig, count_parameters, _BACKEND

if _BACKEND == "mlx":
    import mlx.core as mx
else:
    import torch


def test_default_config() -> None:
    cfg = TokeModelConfig()
    assert cfg.vocab_size == 8192
    assert cfg.hidden_dim == 2048
    assert cfg.num_layers == 24
    assert cfg.num_heads == 16
    assert cfg.num_kv_heads == 4
    assert cfg.ffn_dim == 8192
    assert cfg.max_seq_len == 2048
    assert cfg.dropout == 0.0


def test_parameter_count() -> None:
    """Verify the model has roughly 1B parameters."""
    model = TokeModel(TokeModelConfig())
    n = count_parameters(model)
    # 24 layers x 2048 hidden x 8192 FFN with GQA yields ~1.49B
    assert 1_400_000_000 < n < 1_600_000_000, f"Parameter count {n:,} outside 1.4B-1.6B range"


def test_forward_shape() -> None:
    """Forward pass produces (batch, seq_len, vocab_size) logits."""
    cfg = TokeModelConfig()
    model = TokeModel(cfg)
    batch, seq_len = 1, 64

    if _BACKEND == "mlx":
        tokens = mx.zeros((batch, seq_len), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape == (batch, seq_len, cfg.vocab_size), f"Bad shape: {logits.shape}"
    else:
        tokens = torch.zeros(batch, seq_len, dtype=torch.long)
        logits = model(tokens)
        assert logits.shape == (batch, seq_len, cfg.vocab_size), f"Bad shape: {tuple(logits.shape)}"


def test_custom_vocab_size() -> None:
    """Model works with a non-default vocab size."""
    cfg = TokeModelConfig(vocab_size=16384)
    model = TokeModel(cfg)
    batch, seq_len = 1, 32

    if _BACKEND == "mlx":
        tokens = mx.zeros((batch, seq_len), dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        assert logits.shape[-1] == 16384
    else:
        tokens = torch.zeros(batch, seq_len, dtype=torch.long)
        logits = model(tokens)
        assert logits.shape[-1] == 16384


if __name__ == "__main__":
    test_default_config()
    print("PASS: test_default_config")

    test_parameter_count()
    print("PASS: test_parameter_count")

    test_forward_shape()
    print("PASS: test_forward_shape")

    test_custom_vocab_size()
    print("PASS: test_custom_vocab_size")

    print(f"\nAll tests passed (backend: {_BACKEND})")
