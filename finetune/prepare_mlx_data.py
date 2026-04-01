#!/usr/bin/env python3
"""Convert existing training data to MLX-LM format if needed.

The existing training data (training-data/train.jsonl, training-data/eval.jsonl)
already contains a 'text' field with pre-formatted ChatML strings, which is the
completions format that mlx-lm expects. This script validates the data and
optionally strips extra fields (like 'type') to produce clean MLX-ready JSONL.

In practice, train_mlx.py reads the data directly and handles the format.
This script exists for explicit data preparation and validation.

Usage:
    python prepare_mlx_data.py --input-dir training-data/ --output-dir training-data-mlx/
    python prepare_mlx_data.py --input-dir training-data/ --validate-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_example(record: dict, lineno: int) -> list[str]:
    """Validate a single training example. Returns list of issues."""
    issues: list[str] = []

    if "text" not in record:
        issues.append(f"line {lineno}: missing 'text' field")
        return issues

    text = record["text"]
    if not isinstance(text, str):
        issues.append(f"line {lineno}: 'text' is not a string")
        return issues

    if len(text) == 0:
        issues.append(f"line {lineno}: empty 'text' field")
        return issues

    # Check ChatML structure.
    if "<|im_start|>system" not in text:
        issues.append(f"line {lineno}: missing system turn")
    if "<|im_start|>user" not in text:
        issues.append(f"line {lineno}: missing user turn")
    if "<|im_start|>assistant" not in text:
        issues.append(f"line {lineno}: missing assistant turn")

    return issues


def process_file(input_path: Path, output_path: Path | None) -> tuple[int, int, list[str]]:
    """Process a JSONL file: validate and optionally write clean output.

    Returns (total, valid, issues).
    """
    total = 0
    valid = 0
    all_issues: list[str] = []
    output_records: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                all_issues.append(f"line {lineno}: malformed JSON")
                continue

            issues = validate_example(record, lineno)
            if issues:
                all_issues.extend(issues)
                continue

            valid += 1
            # Output only the 'text' field — MLX-LM completions format.
            output_records.append({"text": record["text"]})

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return total, valid, all_issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Input directory with train.jsonl and eval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for MLX-ready data (omit for validate-only)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate, do not write output")
    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    if args.validate_only:
        args.output_dir = None

    all_ok = True

    for split in ("train", "eval"):
        input_path = args.input_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"  SKIP: {input_path} not found")
            continue

        output_path = None
        if args.output_dir:
            output_path = args.output_dir / f"{split}.jsonl"

        print(f"Processing {input_path}...")
        total, valid, issues = process_file(input_path, output_path)

        print(f"  Total records: {total:,}")
        print(f"  Valid records: {valid:,}")
        if issues:
            all_ok = False
            print(f"  Issues: {len(issues)}")
            # Show first 10 issues.
            for issue in issues[:10]:
                print(f"    WARNING: {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
        else:
            print(f"  No issues found.")

        if output_path:
            print(f"  Written to {output_path}")

    if all_ok:
        print("\nValidation passed. Data is MLX-ready.")
        return 0
    else:
        print("\nValidation found issues. Review warnings above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
