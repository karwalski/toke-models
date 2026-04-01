#!/usr/bin/env python3
"""QLoRA fine-tuning script for toke code generation.

Fine-tunes Qwen 2.5 Coder 7B (or other base models) using QLoRA
on prepared toke corpus training data.

Usage:
    python train_qlora.py --config configs/7b.yaml
    python train_qlora.py --config configs/7b.yaml --resume output/7b-qlora/checkpoint-500
"""
from __future__ import annotations

import argparse
import json
import sys
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
    TrainingArguments,
    Trainer,
)


def load_config(config_path: Path) -> dict:
    """Load YAML training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    model_cfg = config["model"]
    compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def setup_lora(config: dict) -> LoraConfig:
    """Create LoRA configuration."""
    lora_cfg = config["lora"]
    target = lora_cfg.get("target_modules", "all-linear")
    if target == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = target.split(",") if isinstance(target, str) else target

    return LoraConfig(
        r=lora_cfg.get("rank", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )


def setup_training_args(config: dict, resume_from: str | None = None) -> TrainingArguments:
    """Create HuggingFace training arguments."""
    train_cfg = config["training"]
    output_cfg = config["output"]

    return TrainingArguments(
        output_dir=output_cfg["dir"],
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        save_steps=train_cfg.get("save_steps", 500),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 250),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        report_to="none",
        resume_from_checkpoint=resume_from,
    )


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize the dataset using the 'text' field."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    model_name = config["model"]["base"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    print(f"Loading model: {model_name}")
    print(f"  Quantization: {config['model'].get('quantization', 'nf4')}")
    print(f"  LoRA rank: {config['lora'].get('rank', 64)}")
    print(f"  Epochs: {train_cfg.get('epochs', 3)}")
    print(f"  Effective batch: {train_cfg.get('batch_size', 8) * train_cfg.get('gradient_accumulation_steps', 4)}")

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization.
    bnb_config = setup_quantization(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA.
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets.
    print(f"\nLoading training data from {data_cfg['train_file']}...")
    data_files = {"train": data_cfg["train_file"]}
    if Path(data_cfg["eval_file"]).exists():
        data_files["eval"] = data_cfg["eval_file"]

    dataset = load_dataset("json", data_files=data_files)

    # Tokenize.
    print("Tokenizing...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, max_seq_length)
    eval_dataset = None
    if "eval" in dataset:
        eval_dataset = tokenize_dataset(dataset["eval"], tokenizer, max_seq_length)

    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"  Eval examples: {len(eval_dataset)}")

    # Data collator.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments.
    training_args = setup_training_args(config, resume_from=args.resume)

    # Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train.
    print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume)

    # Save adapter.
    adapter_dir = output_cfg.get("adapter_dir", f"{output_cfg['dir']}/adapter")
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved to {adapter_dir}")

    # Save training metrics.
    metrics = train_result.metrics
    metrics_path = Path(output_cfg["dir"]) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Final eval.
    if eval_dataset:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        eval_path = Path(output_cfg["dir"]) / "eval_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Eval metrics saved to {eval_path}")
        print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
