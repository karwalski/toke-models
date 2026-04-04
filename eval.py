#!/usr/bin/env python3
"""Evaluate a trained SentencePiece tokenizer against cl100k_base (tiktoken).

Computes compression ratio, vocabulary utilization, fertility, and
per-program token counts for both the toke tokenizer and baseline.

Usage:
    python eval.py --model models/toke.model --test-data data/valid.txt
    python eval.py --model models/toke.model --test-data data/valid.txt --output eval_report.json
    python eval.py --model models/toke.model --test-data data/valid.txt --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_programs(path: Path) -> list[str]:
    """Load programs from a text file with double-newline separators.

    Matches the write_text() format from prepare.py: programs are
    separated by blank lines (double newlines).
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    # Split on double newlines — each block is one program
    programs = text.split("\n\n")
    # Strip trailing whitespace from each program, drop empty blocks
    return [p.strip() for p in programs if p.strip()]


def tokenize_programs_sp(sp_model: object, programs: list[str]) -> list[list[int]]:
    """Tokenize each program using a SentencePiece model."""
    return [sp_model.encode(p, out_type=int) for p in programs]  # type: ignore[union-attr]


def tokenize_programs_tiktoken(enc: object, programs: list[str]) -> list[list[int]]:
    """Tokenize each program using a tiktoken encoding."""
    return [enc.encode(p) for p in programs]  # type: ignore[union-attr]


def compute_token_stats(token_lists: list[list[int]]) -> dict[str, float]:
    """Compute mean, median, and p95 of token counts per program."""
    if not token_lists:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    counts = [len(toks) for toks in token_lists]
    sorted_counts = sorted(counts)
    p95_idx = int(len(sorted_counts) * 0.95)
    # Clamp index to last element
    p95_idx = min(p95_idx, len(sorted_counts) - 1)
    return {
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "p95": float(sorted_counts[p95_idx]),
    }


def compute_compression_ratio(
    toke_tokens: list[list[int]], baseline_tokens: list[list[int]]
) -> float:
    """Compute compression ratio: total toke tokens / total baseline tokens."""
    toke_total = sum(len(t) for t in toke_tokens)
    baseline_total = sum(len(t) for t in baseline_tokens)
    if baseline_total == 0:
        return 0.0
    return toke_total / baseline_total


def compute_vocab_utilization(
    token_lists: list[list[int]], vocab_size: int
) -> float:
    """Compute vocabulary utilization: unique tokens used / vocab size."""
    if vocab_size == 0:
        return 0.0
    unique = set()
    for toks in token_lists:
        unique.update(toks)
    return len(unique) / vocab_size


def compute_fertility(programs: list[str], token_lists: list[list[int]]) -> float:
    """Compute fertility: mean tokens per character across all programs."""
    if not programs:
        return 0.0
    ratios = []
    for prog, toks in zip(programs, token_lists):
        char_count = len(prog)
        if char_count > 0:
            ratios.append(len(toks) / char_count)
    if not ratios:
        return 0.0
    return statistics.mean(ratios)


def build_report(
    programs: list[str],
    toke_tokens: list[list[int]],
    baseline_tokens: list[list[int]],
    vocab_size: int,
) -> dict:
    """Build the full evaluation report as a dictionary."""
    return {
        "program_count": len(programs),
        "toke": compute_token_stats(toke_tokens),
        "baseline": compute_token_stats(baseline_tokens),
        "compression_ratio": compute_compression_ratio(toke_tokens, baseline_tokens),
        "vocab_utilization": compute_vocab_utilization(toke_tokens, vocab_size),
        "fertility": compute_fertility(programs, toke_tokens),
        "vocab_size": vocab_size,
    }


def format_summary(report: dict) -> str:
    """Format a human-readable summary table from the report."""
    lines = [
        "Tokenizer Evaluation Report",
        "=" * 40,
        f"Programs evaluated:    {report['program_count']}",
        f"Vocabulary size:       {report['vocab_size']}",
        "",
        "Tokens per program         toke     baseline",
        "-" * 40,
        f"  mean                 {report['toke']['mean']:>8.1f}   {report['baseline']['mean']:>8.1f}",
        f"  median               {report['toke']['median']:>8.1f}   {report['baseline']['median']:>8.1f}",
        f"  p95                  {report['toke']['p95']:>8.1f}   {report['baseline']['p95']:>8.1f}",
        "",
        f"Compression ratio:     {report['compression_ratio']:.4f}",
        f"Vocab utilization:     {report['vocab_utilization']:.4f}",
        f"Fertility (tok/char):  {report['fertility']:.4f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a toke SentencePiece tokenizer against cl100k_base."
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="Path to toke .model file"
    )
    parser.add_argument(
        "--test-data", required=True, type=Path, help="Path to test data (valid.txt)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to write JSON report"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print config without running evaluation",
    )
    args = parser.parse_args(argv)

    # Validate inputs
    if not args.model.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        return 1
    if not args.test_data.exists():
        print(f"ERROR: test data not found: {args.test_data}", file=sys.stderr)
        return 1

    # Load programs
    programs = load_programs(args.test_data)
    if not programs:
        print("ERROR: no programs found in test data", file=sys.stderr)
        return 1

    # Dry-run mode
    if args.dry_run:
        print("Dry-run mode — validating inputs only")
        print(f"  Model:     {args.model}")
        print(f"  Test data: {args.test_data}")
        print(f"  Baseline:  {args.baseline}")
        print(f"  Output:    {args.output or '(stdout only)'}")
        print(f"  Programs:  {len(programs)}")
        return 0

    # Load tokenizers (deferred imports so dry-run works without deps)
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: sentencepiece is not installed", file=sys.stderr)
        return 1
    try:
        import tiktoken  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: tiktoken is not installed", file=sys.stderr)
        return 1

    # Load toke model
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.model))
    vocab_size = sp.GetPieceSize()

    # Load baseline
    enc = tiktoken.get_encoding(args.baseline)

    # Tokenize
    toke_tokens = tokenize_programs_sp(sp, programs)
    baseline_tokens = tokenize_programs_tiktoken(enc, programs)

    # Build report
    report = build_report(programs, toke_tokens, baseline_tokens, vocab_size)

    # Output
    summary = format_summary(report)
    print(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
