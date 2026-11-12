#!/usr/bin/env python3
"""Prepare toke corpus for BPE tokenizer training.

Reads JSONL corpus entries, extracts source code, deduplicates,
replaces string literal contents with placeholders, and splits
into train/validation sets.

Usage:
    python prepare.py --input corpus.jsonl --output-dir data/
    python prepare.py --input corpus.jsonl --output-dir data/ --split 0.9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PLACEHOLDER = "<STR>"


def replace_string_literals(source: str) -> str:
    """Replace contents of string literals with a placeholder token.

    Handles escaped quotes inside strings. Replaces "..." with "<STR>".
    """
    result: list[str] = []
    i = 0
    while i < len(source):
        if source[i] == '"':
            result.append('"')
            result.append(PLACEHOLDER)
            i += 1  # skip opening quote
            while i < len(source) and source[i] != '"':
                if source[i] == '\\' and i + 1 < len(source):
                    i += 2  # skip escaped char
                else:
                    i += 1
            if i < len(source):
                result.append('"')
                i += 1  # skip closing quote
        else:
            result.append(source[i])
            i += 1
    return "".join(result)


def deduplicate(sources: list[str]) -> list[str]:
    """Remove exact-duplicate sources using content hashing."""
    seen: set[str] = set()
    unique: list[str] = []
    for src in sources:
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(src)
    return unique


def load_jsonl(path: Path) -> list[str]:
    """Load JSONL file, extract 'tk_source' field from each entry."""
    sources: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed JSON at line {lineno}", file=sys.stderr)
                continue
            source = entry.get("tk_source")
            if not source:
                print(f"WARNING: missing 'tk_source' field at line {lineno}", file=sys.stderr)
                continue
            sources.append(source)
    return sources


def split_data(
    sources: list[str], ratio: float, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split sources into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def write_text(path: Path, sources: list[str]) -> None:
    """Write sources to a text file, one program per double-newline block."""
    with open(path, "w", encoding="utf-8") as f:
        for i, src in enumerate(sources):
            if i > 0:
                f.write("\n\n")
            f.write(src.rstrip("\n"))
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL corpus file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--split", type=float, default=0.9, help="Train split ratio (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    # Load
    sources = load_jsonl(args.input)
    total = len(sources)
    print(f"Loaded {total} entries from {args.input}")

    # Replace string literals
    sources = [replace_string_literals(s) for s in sources]

    # Deduplicate
    sources = deduplicate(sources)
    unique = len(sources)
    print(f"After dedup: {unique} unique entries ({total - unique} duplicates removed)")

    # Split
    train, valid = split_data(sources, args.split, args.seed)
    print(f"Split: {len(train)} train, {len(valid)} validation (ratio={args.split})")

    # Write
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_text(args.output_dir / "train.txt", train)
    write_text(args.output_dir / "valid.txt", valid)
    print(f"Written to {args.output_dir}/{{train,valid}}.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
