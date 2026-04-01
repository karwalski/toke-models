#!/usr/bin/env python3
"""Fuse MLX LoRA adapters into the base model.

After training with train_mlx.py, merges LoRA adapter weights into the
base model to produce a standalone model for inference.

Usage:
    python merge_mlx.py --adapter output/7b-mlx/adapter --output output/7b-mlx/fused
    python merge_mlx.py --adapter output/7b-mlx/adapter --output output/7b-mlx/fused --model Qwen/Qwen2.5-Coder-7B-Instruct
    python merge_mlx.py --adapter output/7b-mlx/adapter --output output/7b-mlx/fused --de-quantize
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import fuse as mlx_fuse


def detect_base_model(adapter_dir: Path) -> str | None:
    """Attempt to detect the base model from adapter training config."""
    # Check training_config.yaml (saved by train_mlx.py).
    config_path = adapter_dir / "training_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        base = cfg.get("model", {}).get("base")
        if base:
            return base

    # Check adapter_config.json (standard mlx-lm format).
    adapter_config = adapter_dir / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            cfg = json.load(f)
        return cfg.get("model", None)

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="Path to saved MLX LoRA adapter directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for fused model")
    parser.add_argument("--model", type=str, default=None,
                        help="Base model name (auto-detected from adapter config if omitted)")
    parser.add_argument("--de-quantize", action="store_true",
                        help="De-quantize the model during fusion (produces float16 weights)")
    args = parser.parse_args(argv)

    if not args.adapter.exists():
        print(f"ERROR: adapter directory not found: {args.adapter}", file=sys.stderr)
        return 1

    # Determine base model.
    if args.model:
        model_name = args.model
    else:
        model_name = detect_base_model(args.adapter)
        if not model_name:
            print(
                "ERROR: cannot determine base model. "
                "Specify --model or ensure training_config.yaml exists in adapter dir.",
                file=sys.stderr,
            )
            return 1

    print(f"Base model: {model_name}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")
    if args.de_quantize:
        print(f"De-quantize: yes")

    # Fuse adapter into base model.
    print("\nFusing adapter into base model...")
    args.output.mkdir(parents=True, exist_ok=True)

    mlx_fuse(
        model=model_name,
        adapter_path=str(args.adapter),
        save_path=str(args.output),
        de_quantize=args.de_quantize,
    )

    print(f"\nFused model saved to {args.output}")

    # List output contents.
    output_files = sorted(args.output.iterdir())
    total_size = sum(f.stat().st_size for f in output_files if f.is_file())
    print(f"  Files: {len(output_files)}")
    print(f"  Total size: {total_size / (1024**3):.1f} GB")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
