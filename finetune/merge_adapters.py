#!/usr/bin/env python3
"""Merge QLoRA adapters back into the base model.

After training, merges LoRA adapter weights into the base model
to produce a standalone model for inference.

Usage:
    python merge_adapters.py --adapter output/7b-qlora/adapter --output output/7b-merged
    python merge_adapters.py --adapter output/7b-qlora/adapter --output output/7b-merged --push-to-hub user/model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for merged model")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name (auto-detected from adapter config if omitted)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="Push merged model to HuggingFace Hub (e.g. user/model-name)")
    args = parser.parse_args(argv)

    if not args.adapter.exists():
        print(f"ERROR: adapter not found: {args.adapter}", file=sys.stderr)
        return 1

    # Detect base model from adapter config.
    adapter_config = args.adapter / "adapter_config.json"
    if args.base_model:
        base_model_name = args.base_model
    elif adapter_config.exists():
        import json
        with open(adapter_config) as f:
            cfg = json.load(f)
        base_model_name = cfg.get("base_model_name_or_path", "")
        if not base_model_name:
            print("ERROR: cannot determine base model from adapter config", file=sys.stderr)
            return 1
    else:
        print("ERROR: no --base-model specified and no adapter_config.json found", file=sys.stderr)
        return 1

    print(f"Base model: {base_model_name}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")

    # Load base model in full precision for merging.
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load and merge adapter.
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model.
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")

    # Push to hub.
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print("Pushed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
