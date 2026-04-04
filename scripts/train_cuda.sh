#!/usr/bin/env bash
# CUDA training script for toke code generation fine-tuning.
# Story 10.7.9: Hybrid MLX+CUDA training infrastructure
#
# Uses the same YAML config format and JSONL data pipeline as the MLX trainer.
# Supports QLoRA and DoRA adapters via PEFT/transformers on NVIDIA GPUs.
#
# Usage:
#   ./scripts/train_cuda.sh --config finetune/configs/7b.yaml
#   ./scripts/train_cuda.sh --config finetune/configs/7b.yaml --output-dir output/cuda-run1
#   ./scripts/train_cuda.sh --config finetune/configs/7b.yaml --dry-run
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - Python 3.10+ with torch, transformers, peft, bitsandbytes, datasets, pyyaml
#   - Optional: wandb (pip install wandb) for experiment tracking

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

# ─── Defaults ────────────────────────────────────────────────────────────────

CONFIG=""
OUTPUT_DIR=""
DRY_RUN=false
RESUME=""

# ─── Parse arguments ────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

CUDA fine-tuning for toke models using the same config format as MLX.

Options:
  --config PATH        Path to YAML config file (required)
  --output-dir PATH    Override output directory from config
  --dry-run            Validate config and data, print plan, exit without training
  --resume PATH        Resume from checkpoint directory
  --help, -h           Show this help

Environment:
  WANDB_PROJECT        W&B project name (default: toke-training)
  WANDB_DISABLED       Set to "true" to disable W&B logging
  CUDA_VISIBLE_DEVICES GPU selection (e.g., "0" or "0,1")

Examples:
  # Standard QLoRA training
  $(basename "$0") --config finetune/configs/7b.yaml

  # DoRA training with custom output
  $(basename "$0") --config finetune/configs/7b_mlx_dora.yaml --output-dir output/dora-cuda

  # Dry run to validate config
  $(basename "$0") --config finetune/configs/7b.yaml --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --resume)     RESUME="$2"; shift 2 ;;
        --help|-h)    usage; exit 0 ;;
        *)            echo "ERROR: Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required" >&2
    usage
    exit 1
fi

# Resolve config path relative to repo root if not absolute.
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$REPO_DIR/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# ─── Environment check ──────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/cuda_training_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== CUDA Training Pipeline ==="
log "Config:    $CONFIG"
log "Log:       $LOG_FILE"
log "Dry run:   $DRY_RUN"

# Check for CUDA availability.
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    log "WARNING: CUDA not available. Training will fail or fall back to CPU."
    log "  Ensure NVIDIA drivers and PyTorch with CUDA support are installed."
    if ! $DRY_RUN; then
        log "ERROR: Cannot proceed without CUDA. Use --dry-run to validate config only."
        exit 1
    fi
else
    GPU_INFO=$(python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1024**3:.1f} GB)')
" 2>/dev/null || echo "  (could not query GPU)")
    log "CUDA GPUs detected:"
    log "$GPU_INFO"
fi

# Check required Python packages.
MISSING_PKGS=""
for pkg in torch transformers peft datasets bitsandbytes accelerate yaml; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [[ -n "$MISSING_PKGS" ]]; then
    log "ERROR: Missing Python packages:$MISSING_PKGS"
    log "Install with: pip install torch transformers peft datasets bitsandbytes accelerate pyyaml"
    exit 1
fi

# ─── Parse config and display training plan ──────────────────────────────────

PLAN=$(python3 - "$CONFIG" "$OUTPUT_DIR" <<'PYEOF'
import sys, yaml, json
from pathlib import Path

config_path = sys.argv[1]
output_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else ""

with open(config_path) as f:
    config = yaml.safe_load(f)

model_cfg = config.get("model", {})
lora_cfg = config.get("lora", {})
train_cfg = config.get("training", {})
data_cfg = config.get("data", {})
output_cfg = config.get("output", {})

# Determine adapter type: DoRA if use_dora is set, QLoRA otherwise.
use_dora = lora_cfg.get("use_dora", False)
adapter_type = "DoRA" if use_dora else "QLoRA"

# Output directory.
if output_override:
    output_dir = output_override
else:
    output_dir = output_cfg.get("dir", output_cfg.get("adapter_dir", "output/cuda"))

batch_size = train_cfg.get("batch_size", 8)
grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
effective_batch = batch_size * grad_accum

plan = {
    "model": model_cfg.get("base", "unknown"),
    "adapter_type": adapter_type,
    "use_dora": use_dora,
    "quantization": model_cfg.get("quantization", "nf4"),
    "rank": lora_cfg.get("rank", 64),
    "alpha": lora_cfg.get("alpha", 128),
    "dropout": lora_cfg.get("dropout", 0.05),
    "epochs": train_cfg.get("epochs", 3),
    "batch_size": batch_size,
    "grad_accum": grad_accum,
    "effective_batch": effective_batch,
    "learning_rate": train_cfg.get("learning_rate", 2e-4),
    "lr_scheduler": train_cfg.get("lr_scheduler", train_cfg.get("lr_schedule", "cosine")),
    "warmup_steps": train_cfg.get("warmup_steps", 100),
    "max_seq_length": train_cfg.get("max_seq_length", 2048),
    "bf16": train_cfg.get("bf16", True),
    "gradient_checkpointing": train_cfg.get("gradient_checkpointing", train_cfg.get("grad_checkpoint", True)),
    "train_file": data_cfg.get("train_file", "training-data/train.jsonl"),
    "eval_file": data_cfg.get("eval_file", "training-data/eval.jsonl"),
    "output_dir": output_dir,
    "save_steps": train_cfg.get("save_steps", train_cfg.get("save_every", 500)),
    "logging_steps": train_cfg.get("logging_steps", train_cfg.get("steps_per_report", 10)),
    "eval_steps": train_cfg.get("eval_steps", train_cfg.get("steps_per_eval", 250)),
    "train_embeddings": train_cfg.get("train_embeddings", False),
}

print(json.dumps(plan))
PYEOF
)

if [[ -z "$PLAN" ]]; then
    log "ERROR: Failed to parse config"
    exit 1
fi

# Extract plan values for display.
MODEL=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model'])")
ADAPTER_TYPE=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['adapter_type'])")
RANK=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['rank'])")
ALPHA=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['alpha'])")
EPOCHS=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['epochs'])")
EFF_BATCH=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['effective_batch'])")
TRAIN_FILE=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['train_file'])")
EVAL_FILE=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['eval_file'])")
PLAN_OUTPUT_DIR=$(echo "$PLAN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['output_dir'])")

log ""
log "Training plan:"
log "  Model:           $MODEL"
log "  Adapter:         $ADAPTER_TYPE"
log "  Rank:            $RANK"
log "  Alpha:           $ALPHA"
log "  Epochs:          $EPOCHS"
log "  Effective batch: $EFF_BATCH"
log "  Train data:      $TRAIN_FILE"
log "  Eval data:       $EVAL_FILE"
log "  Output:          $PLAN_OUTPUT_DIR"

# Verify training data exists.
TRAIN_DATA_PATH="$REPO_DIR/$TRAIN_FILE"
if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    # Try absolute path.
    if [[ -f "$TRAIN_FILE" ]]; then
        TRAIN_DATA_PATH="$TRAIN_FILE"
    else
        log "ERROR: Training data not found: $TRAIN_FILE"
        exit 1
    fi
fi

TRAIN_LINES=$(wc -l < "$TRAIN_DATA_PATH" | tr -d ' ')
log "  Train examples:  $TRAIN_LINES"

EVAL_DATA_PATH="$REPO_DIR/$EVAL_FILE"
if [[ -f "$EVAL_DATA_PATH" ]]; then
    EVAL_LINES=$(wc -l < "$EVAL_DATA_PATH" | tr -d ' ')
    log "  Eval examples:   $EVAL_LINES"
elif [[ -f "$EVAL_FILE" ]]; then
    EVAL_LINES=$(wc -l < "$EVAL_FILE" | tr -d ' ')
    log "  Eval examples:   $EVAL_LINES"
else
    log "  Eval data:       (not found, skipping evaluation)"
fi

if $DRY_RUN; then
    log ""
    log "=== DRY RUN — config and data validated, no training performed ==="
    exit 0
fi

# ─── Run training ────────────────────────────────────────────────────────────

log ""
log "Starting CUDA training..."

# Build Python command-line arguments.
TRAIN_ARGS=(
    "--config" "$CONFIG"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    TRAIN_ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS+=("--resume" "$RESUME")
fi

# Check wandb availability.
WANDB_AVAILABLE=false
if python3 -c "import wandb" 2>/dev/null; then
    if [[ "${WANDB_DISABLED:-false}" != "true" ]]; then
        WANDB_AVAILABLE=true
        export WANDB_PROJECT="${WANDB_PROJECT:-toke-training}"
        log "W&B logging enabled (project: $WANDB_PROJECT)"
    fi
fi

cd "$REPO_DIR"

python3 - "${TRAIN_ARGS[@]}" <<'TRAIN_PYEOF'
"""CUDA training driver for toke models.

Reads the same YAML config format as train_mlx.py but uses
PEFT/transformers with BitsAndBytes quantization for NVIDIA GPUs.
Supports QLoRA and DoRA adapters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: Path) -> dict:
    """Load YAML training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes 4-bit quantization config."""
    model_cfg = config.get("model", {})
    compute_dtype = getattr(
        torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    )
    quant_type = model_cfg.get("quantization", "nf4")
    # MLX configs use quantization: true — map to nf4 for CUDA.
    if quant_type is True or quant_type == "true":
        quant_type = "nf4"

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def setup_lora(config: dict) -> LoraConfig:
    """Create LoRA/DoRA configuration from shared YAML config.

    Handles both CUDA-native configs (target_modules, task_type) and
    MLX-style configs (keys, use_dora) transparently.
    """
    lora_cfg = config.get("lora", {})

    # Target modules: CUDA uses target_modules, MLX uses keys.
    target = lora_cfg.get("target_modules", None)
    if target is None:
        # Map MLX 'keys' to PEFT target_modules.
        keys = lora_cfg.get("keys", None)
        if keys:
            target = keys
        else:
            target = "all-linear"

    if isinstance(target, str) and target != "all-linear":
        target = target.split(",")

    use_dora = lora_cfg.get("use_dora", False)

    return LoraConfig(
        r=lora_cfg.get("rank", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target,
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        use_dora=use_dora,
    )


def setup_training_args(
    config: dict,
    output_dir: str,
    resume_from: str | None = None,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from shared YAML config.

    Handles both CUDA-native keys and MLX-style keys.
    """
    train_cfg = config.get("training", {})

    # Determine W&B reporting.
    report_to = "none"
    try:
        import wandb  # noqa: F401

        if os.environ.get("WANDB_DISABLED", "false").lower() != "true":
            report_to = "wandb"
    except ImportError:
        pass

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg.get(
            "gradient_accumulation_steps", 4
        ),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get(
            "lr_scheduler", train_cfg.get("lr_schedule", "cosine")
        ),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get(
            "gradient_checkpointing", train_cfg.get("grad_checkpoint", True)
        ),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        save_steps=train_cfg.get(
            "save_steps", train_cfg.get("save_every", 500)
        ),
        logging_steps=train_cfg.get(
            "logging_steps", train_cfg.get("steps_per_report", 10)
        ),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get(
            "eval_steps", train_cfg.get("steps_per_eval", 250)
        ),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        report_to=report_to,
        resume_from_checkpoint=resume_from,
    )


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize using the 'text' field (same JSONL format as MLX pipeline)."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA training for toke models")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["base"]
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    lora_cfg = config.get("lora", {})
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    # Determine output directory.
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = output_cfg.get("dir", output_cfg.get("adapter_dir", "output/cuda"))

    adapter_dir = os.path.join(output_dir, "adapter")

    use_dora = lora_cfg.get("use_dora", False)
    adapter_type = "DoRA" if use_dora else "QLoRA"

    print(f"CUDA {adapter_type} fine-tuning")
    print(f"  Model: {model_name}")
    print(f"  Adapter: {adapter_type}")
    print(f"  Rank: {lora_cfg.get('rank', 64)}")
    print(f"  Alpha: {lora_cfg.get('alpha', 128)}")
    print(f"  Epochs: {train_cfg.get('epochs', 3)}")
    eff_batch = train_cfg.get("batch_size", 8) * train_cfg.get(
        "gradient_accumulation_steps", 4
    )
    print(f"  Effective batch: {eff_batch}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization.
    print(f"\nLoading model with 4-bit quantization...")
    bnb_config = setup_quantization(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA/DoRA adapters.
    print(f"Applying {adapter_type} adapters...")
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optionally unfreeze embedding layers (for custom tokenizer support).
    if train_cfg.get("train_embeddings", False):
        print("Unfreezing embed_tokens and lm_head for custom tokenizer...")
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = True

    # Load datasets (same JSONL format as MLX pipeline).
    print(f"\nLoading training data from {data_cfg['train_file']}...")
    data_files = {"train": data_cfg["train_file"]}
    eval_file = data_cfg.get("eval_file", "")
    if eval_file and Path(eval_file).exists():
        data_files["eval"] = eval_file

    dataset = load_dataset("json", data_files=data_files)

    # Tokenize.
    print("Tokenizing...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, max_seq_length)
    eval_dataset = None
    if "eval" in dataset:
        eval_dataset = tokenize_dataset(dataset["eval"], tokenizer, max_seq_length)

    print(f"  Train examples: {len(train_dataset):,}")
    if eval_dataset:
        print(f"  Eval examples: {len(eval_dataset):,}")

    # Data collator.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments.
    training_args = setup_training_args(config, output_dir, resume_from=args.resume)

    # Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train.
    print(f"\nStarting {adapter_type} training...")
    print(f"  Output: {output_dir}")
    print(f"  Checkpoints saved every {training_args.save_steps} steps")
    start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=args.resume)

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining complete in {hours:.1f} hours ({elapsed:.0f}s)")

    # Save adapter.
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    # Save training metrics.
    metrics = train_result.metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save training summary (same format as MLX trainer).
    summary = {
        "model": model_name,
        "adapter_type": "dora" if use_dora else "qlora",
        "framework": "cuda",
        "lora_rank": lora_cfg.get("rank", 64),
        "lora_alpha": lora_cfg.get("alpha", 128),
        "epochs": train_cfg.get("epochs", 3),
        "batch_size": train_cfg.get("batch_size", 8),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 4),
        "learning_rate": train_cfg.get("learning_rate", 2e-4),
        "training_time_seconds": elapsed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset else 0,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1),
    }
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Save config for reproducibility.
    config_save_path = Path(output_dir) / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Final evaluation.
    if eval_dataset:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        eval_path = Path(output_dir) / "eval_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
        print(f"  Eval metrics saved to {eval_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
TRAIN_PYEOF

EXIT_CODE=$?

log ""
if [[ $EXIT_CODE -eq 0 ]]; then
    log "=== CUDA training pipeline complete ==="
    log "Training log: $LOG_FILE"
    log "Output: $PLAN_OUTPUT_DIR"
else
    log "=== CUDA training FAILED (exit code: $EXIT_CODE) ==="
    log "Check log: $LOG_FILE"
fi

exit $EXIT_CODE
