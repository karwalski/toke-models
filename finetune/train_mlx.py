#!/usr/bin/env python3
"""MLX LoRA/DoRA fine-tuning script for toke code generation on Apple Silicon.

Fine-tunes Qwen 2.5 Coder 7B (or other base models) using LoRA or DoRA with MLX
on prepared toke corpus training data. Designed for Mac Studio M4 Max.

Usage:
    python train_mlx.py --config configs/7b_mlx.yaml
    python train_mlx.py --config configs/7b_mlx_dora.yaml
    python train_mlx.py --config configs/7b_mlx.yaml --resume output/7b-mlx/adapter
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import yaml
from mlx_lm import load as mlx_load
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
from mlx_lm.tuner.utils import linear_to_lora_layers


def load_config(config_path: Path) -> dict:
    """Load YAML training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_lora_config(config: dict) -> dict:
    """Build LoRA/DoRA layer configuration dict for mlx-lm."""
    lora_cfg = config["lora"]
    result: dict = {
        "rank": lora_cfg.get("rank", 64),
        "alpha": lora_cfg.get("alpha", 128.0),
        "dropout": lora_cfg.get("dropout", 0.05),
        "scale": lora_cfg.get("alpha", 128.0) / lora_cfg.get("rank", 64),
    }
    if "keys" in lora_cfg:
        result["keys"] = lora_cfg["keys"]
    if lora_cfg.get("use_dora", False):
        result["use_dora"] = True
    return result


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    trainable = sum(
        p.size for _, p in mlx.utils.tree_flatten(model.trainable_parameters())
    )
    return total, trainable


def setup_training_args(config: dict, adapter_dir: str) -> TrainingArgs:
    """Create MLX TrainingArgs from config."""
    train_cfg = config["training"]

    # Calculate iterations from epochs if training data size is known.
    # mlx_train uses iters (number of steps), not epochs directly.
    # We set iters=0 here and let the caller compute from data size.
    return TrainingArgs(
        batch_size=train_cfg.get("batch_size", 8),
        iters=0,  # Computed from epochs and data size below.
        val_batches=25,
        steps_per_report=train_cfg.get("steps_per_report", 10),
        steps_per_eval=train_cfg.get("steps_per_eval", 250),
        steps_per_save=train_cfg.get("save_every", 500),
        adapter_file=str(Path(adapter_dir) / "adapters.safetensors"),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        grad_checkpoint=train_cfg.get("grad_checkpoint", True),
        grad_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
    )


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL data as a list of dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from adapter directory")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    model_name = config["model"]["base"]
    train_cfg = config["training"]
    output_cfg = config["output"]
    adapter_dir = output_cfg.get("adapter_dir", "output/7b-mlx/adapter")
    lora_cfg = build_lora_config(config)

    use_dora_flag = lora_cfg.get("use_dora", False)
    adapter_label = "DoRA" if use_dora_flag else "LoRA"
    print(f"MLX {adapter_label} fine-tuning")
    print(f"  Model: {model_name}")
    print(f"  Quantized: {config['model'].get('quantization', True)}")
    print(f"  Adapter: {adapter_label}")
    print(f"  Rank: {lora_cfg['rank']}")
    print(f"  Alpha: {lora_cfg['alpha']}")
    print(f"  Epochs: {train_cfg.get('epochs', 3)}")
    eff_batch = train_cfg.get("batch_size", 8) * train_cfg.get("gradient_accumulation_steps", 4)
    print(f"  Effective batch: {eff_batch}")

    # Load model and tokenizer.
    print(f"\nLoading model: {model_name}...")
    model, tokenizer = mlx_load(model_name)

    # Apply LoRA/DoRA layers.
    use_dora = lora_cfg.pop("use_dora", False)
    adapter_type = "DoRA" if use_dora else "LoRA"
    print(f"Applying {adapter_type} adapters...")
    # num_layers controls how many transformer blocks (from the end) get
    # adapter layers.  Default large value (999) means "all blocks".
    num_lora_layers = config["lora"].get("num_layers", 999)
    linear_to_lora_layers(
        model,
        num_layers=num_lora_layers,
        config=lora_cfg,
        use_dora=use_dora,
    )

    # Unfreeze embed_tokens and lm_head for custom tokenizer support.
    # When using a custom BPE tokenizer with tokens absent from the base
    # model's vocabulary, these layers MUST be fully trainable — otherwise
    # the model produces random outputs for Toke-specific tokens.
    # New token embeddings are initialized by MLX's default (zero or random);
    # averaging sub-word embeddings is handled during tokenizer setup.
    if config.get("training", {}).get("train_embeddings", True):
        print("Unfreezing embed_tokens and lm_head for custom tokenizer...")
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens.unfreeze()
        if hasattr(model, "lm_head"):
            model.lm_head.unfreeze()

    # Load existing adapter weights if resuming.
    if args.resume:
        resume_path = Path(args.resume)
        adapter_file = resume_path / "adapters.safetensors"
        if adapter_file.exists():
            print(f"Resuming from {adapter_file}...")
            model.load_weights(str(adapter_file), strict=False)
        else:
            print(f"WARNING: no adapters.safetensors in {resume_path}", file=sys.stderr)

    # Report parameters.
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Load data and create tokenized datasets.
    # Data uses OpenAI messages format: {"messages": [{role, content}, ...]}
    print(f"\nLoading training data...")
    data_cfg = config["data"]
    train_path = Path(data_cfg["train_file"])
    eval_path = Path(data_cfg["eval_file"])
    if not train_path.exists():
        print(f"ERROR: training data not found: {train_path}", file=sys.stderr)
        return 1
    train_raw = load_jsonl(train_path)
    eval_raw = load_jsonl(eval_path) if eval_path.exists() else []
    print(f"  Train examples: {len(train_raw):,}")
    print(f"  Eval examples: {len(eval_raw):,}")

    # Wrap in ChatDataset for tokenization.
    train_data = ChatDataset(train_raw, tokenizer, chat_key="messages")
    eval_data = ChatDataset(eval_raw, tokenizer, chat_key="messages") if eval_raw else None

    # Compute iterations from epochs.
    batch_size = train_cfg.get("batch_size", 8)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    epochs = train_cfg.get("epochs", 3)
    steps_per_epoch = math.ceil(len(train_data) / (batch_size * grad_accum))
    total_iters = steps_per_epoch * epochs
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total iterations: {total_iters:,}")

    # Build training args.
    training_args = setup_training_args(config, adapter_dir)
    training_args.iters = total_iters

    # Create adapter output directory.
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)

    # Save config alongside adapter for reproducibility.
    config_save_path = Path(adapter_dir) / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Build optimizer.
    lr = train_cfg.get("learning_rate", 2e-4)
    warmup_steps = train_cfg.get("warmup_steps", 100)

    # Cosine schedule with warmup.
    warmup = optim.linear_schedule(
        init=1e-7,
        end=lr,
        steps=warmup_steps,
    )
    cosine = optim.cosine_decay(
        init=lr,
        decay_steps=total_iters - warmup_steps,
    )
    lr_schedule = optim.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # Train.
    print(f"\nStarting training...")
    print(f"  Adapter output: {adapter_dir}")
    start_time = time.time()

    mlx_train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=train_data,
        val_dataset=eval_data,
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining complete in {hours:.1f} hours ({elapsed:.0f}s)")

    # Save final adapter.
    final_adapter = Path(adapter_dir) / "adapters.safetensors"
    if final_adapter.exists():
        print(f"Adapter saved to {adapter_dir}")
    else:
        # mlx_train saves automatically, but log if missing.
        print(f"WARNING: expected adapter file not found at {final_adapter}", file=sys.stderr)

    # Save training summary.
    summary = {
        "model": model_name,
        "adapter_type": "dora" if use_dora else "lora",
        "lora_rank": lora_cfg["rank"],
        "lora_alpha": lora_cfg["alpha"],
        "epochs": epochs,
        "total_iters": total_iters,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "training_time_seconds": elapsed,
        "train_examples": len(train_raw),
        "eval_examples": len(eval_raw),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    summary_path = Path(adapter_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
