#!/usr/bin/env python3
"""BPE training wrapper for the toke tokenizer.

Wraps SentencePiece for BPE training on toke corpus data prepared
by prepare.py.

Usage:
    python train.py --input data/train.txt --output-dir models/
    python train.py --input data/train.txt --output-dir models/ --vocab-size 4000
    python train.py --input data/train.txt --output-dir models/ --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Toke-specific defaults for SentencePiece training.
DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MODEL_TYPE = "bpe"
# Toke uses only 80 ASCII chars, so full coverage is correct.
CHARACTER_COVERAGE = 1.0
# Placeholder token injected by prepare.py for string literal contents.
USER_DEFINED_SYMBOLS = ["<STR>"]
# Special token IDs.
PAD_ID = -1
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2


def build_config(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict:
    """Build the SentencePiece training configuration dict."""
    model_prefix = str(output_dir / "toke")
    return {
        "input": str(input_path),
        "model_prefix": model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": CHARACTER_COVERAGE,
        "user_defined_symbols": USER_DEFINED_SYMBOLS,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        # Code-specific: disable natural-language normalization defaults.
        # toke uses 80 ASCII chars — no Unicode normalization needed.
        "normalization_rule_name": "identity",
        "add_dummy_prefix": False,
        "remove_extra_whitespaces": False,
        # Ensure all byte values can be represented (newlines, etc.)
        "byte_fallback": True,
    }


def print_config(config: dict) -> None:
    """Print the training configuration in a human-readable format."""
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def count_special_tokens() -> int:
    """Return the number of special/reserved tokens."""
    # unk, bos, eos, plus user-defined symbols
    count = 3  # unk, bos, eos (pad_id=-1 means no pad token)
    count += len(USER_DEFINED_SYMBOLS)
    return count


def print_vocab_stats(vocab_size: int) -> None:
    """Print vocabulary statistics after training."""
    special = count_special_tokens()
    print("Vocab stats:")
    print(f"  Total vocab size: {vocab_size}")
    print(f"  Special tokens:   {special}")
    print(f"  BPE merges:       {vocab_size - special}")


def train(config: dict) -> int:
    """Run SentencePiece training with the given configuration.

    Returns 0 on success, 1 on error.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print(
            "ERROR: sentencepiece is not installed. "
            "Install it with: pip install sentencepiece",
            file=sys.stderr,
        )
        return 1

    # SentencePiece expects user_defined_symbols as a comma-separated string.
    sp_config = dict(config)
    sp_config["user_defined_symbols"] = ",".join(config["user_defined_symbols"])

    try:
        spm.SentencePieceTrainer.train(**sp_config)
    except Exception as exc:
        print(f"ERROR: training failed: {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BPE training wrapper for the toke tokenizer."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input training text file (output of prepare.py)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save toke.model and toke.vocab",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["bpe", "unigram", "char", "word"],
        help=f"SentencePiece model type (default: {DEFAULT_MODEL_TYPE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and print config without training",
    )
    args = parser.parse_args(argv)

    # Validate input file exists.
    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"ERROR: input path is not a file: {args.input}", file=sys.stderr)
        return 1

    # Validate vocab size.
    if args.vocab_size < 1:
        print(f"ERROR: vocab-size must be positive, got {args.vocab_size}", file=sys.stderr)
        return 1

    # Build config.
    config = build_config(
        input_path=args.input,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )

    print_config(config)

    if args.dry_run:
        print("\n[dry-run] Skipping training.")
        return 0

    # Ensure output directory exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train.
    print(f"\nTraining {args.model_type} model with vocab_size={args.vocab_size}...")
    rc = train(config)
    if rc != 0:
        return rc

    # Verify output files were created.
    model_path = args.output_dir / "toke.model"
    vocab_path = args.output_dir / "toke.vocab"
    if not model_path.exists():
        print(f"ERROR: expected model file not found: {model_path}", file=sys.stderr)
        return 1
    if not vocab_path.exists():
        print(f"ERROR: expected vocab file not found: {vocab_path}", file=sys.stderr)
        return 1

    print(f"\nModel saved to {model_path}")
    print(f"Vocab saved to {vocab_path}")
    print_vocab_stats(args.vocab_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
