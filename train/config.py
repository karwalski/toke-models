"""Training configuration for the toke 1B model.

Hyperparameters derived from the training infrastructure design:
  docs/architecture/training-infrastructure.md

Usage:
    from train.config import TrainConfig
    cfg = TrainConfig()
    cfg = TrainConfig(batch_size=64, total_steps=100_000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """All hyperparameters for a single training run."""

    # ── Model architecture ──────────────────────────────────────────
    vocab_size: int = 8192
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4          # grouped-query attention
    ffn_dim: int = 8192
    max_seq_len: int = 2048

    # ── Training ────────────────────────────────────────────────────
    batch_size: int = 32
    micro_batch_size: int = 4
    gradient_accumulation: int = 8  # batch_size / micro_batch_size
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    total_steps: int = 50_000
    weight_decay: float = 0.1
    gradient_clip: float = 1.0

    # ── Infrastructure ──────────────────────────────────────────────
    precision: str = "bfloat16"
    checkpoint_every: int = 1_000
    eval_every: int = 5_000
    keep_checkpoints: int = 5

    # ── Paths (relative to repo root) ──────────────────────────────
    corpus_path: str = "corpus/filtered/filtered.jsonl"
    tokenizer_path: str = "tokenizer/vocab/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"

    # ── Derived helpers ─────────────────────────────────────────────
    def resolve_paths(self, root: Path | str = ".") -> dict[str, Path]:
        """Return absolute Path objects for every configured path."""
        root = Path(root).resolve()
        return {
            "corpus": root / self.corpus_path,
            "tokenizer": root / self.tokenizer_path,
            "checkpoints": root / self.checkpoint_dir,
            "logs": root / self.log_dir,
        }
