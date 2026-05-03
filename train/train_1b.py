"""Main training script for the toke 1B model.

Designed for MLX on Apple Silicon (Mac Studio M4 Max).
PyTorch/DeepSpeed fallback for cloud GPU runs is a future extension.

Usage:
    python -m train.train_1b
    python -m train.train_1b --total-steps 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from train.config import TrainConfig

logger = logging.getLogger(__name__)


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train toke 1B model")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Override total training steps")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--root", type=str, default=".",
                   help="Repository root directory")
    return p.parse_args()


# ── Tokenizer ───────────────────────────────────────────────────────

def load_tokenizer(path: Path):
    """Load trained BPE tokenizer from disk.

    TODO: Load tokenizers.Tokenizer from ``path / tokenizer.json``
    and return it.
    """
    raise NotImplementedError("Tokenizer loading not yet implemented")


# ── Data ────────────────────────────────────────────────────────────

def create_data_loader(corpus_path: Path, tokenizer, config: TrainConfig):
    """Create a streaming data loader over the JSONL corpus.

    TODO:
      - Stream lines from *corpus_path*
      - Tokenize on-the-fly with *tokenizer*
      - Yield batches of shape (micro_batch_size, max_seq_len)
      - Handle shuffling and epoch boundaries
    """
    raise NotImplementedError("Data loader not yet implemented")


# ── Model ───────────────────────────────────────────────────────────

def build_model(config: TrainConfig):
    """Instantiate the 1B decoder-only transformer.

    TODO:
      - Import model definition from train.model (story 81.3b)
      - Initialise weights with appropriate scheme
      - Return (model, param_count)
    """
    raise NotImplementedError("Model definition not yet implemented")


# ── Optimizer ───────────────────────────────────────────────────────

def build_optimizer(model, config: TrainConfig):
    """Create AdamW optimizer with cosine learning rate schedule.

    TODO:
      - Use mlx.optimizers.AdamW
      - Attach cosine schedule: warmup_steps -> peak lr -> cosine decay to min_lr
      - Apply weight_decay to non-bias, non-layernorm params
    """
    raise NotImplementedError("Optimizer not yet implemented")


# ── Checkpointing ──────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step: int, config: TrainConfig,
                    checkpoint_dir: Path) -> Path:
    """Save model weights, optimizer state, and training step.

    TODO:
      - Save to checkpoint_dir / f"step_{step:06d}.safetensors"
      - Prune old checkpoints to keep only config.keep_checkpoints
    """
    raise NotImplementedError("Checkpoint saving not yet implemented")


def load_checkpoint(path: Path, model, optimizer):
    """Resume training from a saved checkpoint.

    TODO:
      - Load model weights and optimizer state
      - Return the step number to resume from
    """
    raise NotImplementedError("Checkpoint loading not yet implemented")


# ── Logging ─────────────────────────────────────────────────────────

def init_logging(log_dir: Path) -> Path:
    """Set up JSONL training log.

    TODO:
      - Create log_dir / "train.jsonl"
      - Write header row with config summary
      - Return log file path
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    return log_path


def log_step(log_path: Path, step: int, loss: float, lr: float,
             grad_norm: float, elapsed: float) -> None:
    """Append a single training step record to the JSONL log."""
    record = {
        "step": step,
        "loss": loss,
        "lr": lr,
        "grad_norm": grad_norm,
        "elapsed_s": round(elapsed, 3),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Training loop ──────────────────────────────────────────────────

def train(config: TrainConfig, root: Path, resume_path: str | None = None):
    """Main training entry point."""
    paths = config.resolve_paths(root)

    # 1. Tokenizer
    logger.info("Loading tokenizer from %s", paths["tokenizer"])
    # TODO: tokenizer = load_tokenizer(paths["tokenizer"])

    # 2. Data loader
    logger.info("Creating data loader from %s", paths["corpus"])
    # TODO: loader = create_data_loader(paths["corpus"], tokenizer, config)

    # 3. Model
    logger.info("Building 1B model")
    # TODO: model, n_params = build_model(config)
    # logger.info("Model has %s parameters", f"{n_params:,}")

    # 4. Optimizer
    # TODO: optimizer = build_optimizer(model, config)

    # 5. Resume
    start_step = 0
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        # TODO: start_step = load_checkpoint(Path(resume_path), model, optimizer)

    # 6. Logging
    log_path = init_logging(paths["logs"])
    logger.info("Logging to %s", log_path)

    # 7. Training loop
    logger.info("Starting training from step %d to %d", start_step, config.total_steps)
    t0 = time.time()

    for step in range(start_step, config.total_steps):
        # TODO: Implement forward/backward pass
        #   batch = next(loader)
        #   loss, grad_norm = train_step(model, optimizer, batch)
        #   lr = get_current_lr(optimizer, step)
        #   log_step(log_path, step, loss, lr, grad_norm, time.time() - t0)
        pass

        # Checkpoint
        if (step + 1) % config.checkpoint_every == 0:
            # TODO: save_checkpoint(model, optimizer, step + 1, config, paths["checkpoints"])
            logger.info("Checkpoint saved at step %d", step + 1)

        # Evaluation
        if (step + 1) % config.eval_every == 0:
            # TODO: Run Pass@1 on 100-task subset (see eval_during_train.py)
            logger.info("Evaluation at step %d (not yet implemented)", step + 1)

    logger.info("Training complete at step %d (%.1f hours)",
                config.total_steps, (time.time() - t0) / 3600)


# ── Entry point ─────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = TrainConfig()
    if args.total_steps is not None:
        config.total_steps = args.total_steps

    root = Path(args.root).resolve()
    train(config, root, resume_path=args.resume)


if __name__ == "__main__":
    main()
