#!/usr/bin/env python3
"""Prepare toke corpus for QLoRA fine-tuning.

Converts corpus.jsonl entries into instruction-tuning format per RFC Section 15.3.
Produces three example types:
  1. Direct generation — prompt -> toke code
  2. Correction — broken code + error -> fixed code
  3. Multi-language comparison — side-by-side with token counts

Usage:
    python prepare_data.py --corpus ../tokenizer/corpus.jsonl --output-dir training-data/
    python prepare_data.py --corpus corpus.jsonl --output-dir training-data/ --split 0.95
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Task category descriptions for generating natural-language prompts.
CATEGORY_DESC: dict[str, str] = {
    "A-ARR": "array manipulation",
    "A-CND": "conditional logic and branching",
    "A-ERR": "error handling with error unions",
    "A-MTH": "mathematical computation",
    "A-SRT": "sorting and searching",
    "A-STR": "string processing",
    "B-CMP": "multi-function composition",
    "C-EDG": "edge case handling",
    "D-APP": "application-level program",
}


def task_id_to_prompt(task_id: str, tk_source: str) -> str:
    """Generate a natural-language instruction from task metadata and source."""
    parts = task_id.split("-")
    cat = "-".join(parts[:2]) if len(parts) >= 2 else task_id
    cat_desc = CATEGORY_DESC.get(cat, "general programming")

    # Extract function name and signature from source.
    func_sig = ""
    for line in tk_source.split(";"):
        line = line.strip()
        if line.startswith("F=") or line.startswith("f="):
            sig_part = line[2:]
            paren = sig_part.find("(")
            brace = sig_part.find("{")
            if paren > 0 and brace > paren:
                func_sig = sig_part[:brace]
            break

    if func_sig:
        return (
            f"Write a toke function `{func_sig}` that performs {cat_desc}. "
            f"Use only toke syntax (lp for loops, el for else, < for return, "
            f"; as separator in parameters)."
        )
    return (
        f"Write a toke program for {cat_desc}. "
        f"Use only toke syntax (lp for loops, el for else, < for return)."
    )


def make_direct_example(entry: dict) -> dict:
    """Create a direct generation training example."""
    tk_source = entry["tk_source"]
    prompt = task_id_to_prompt(entry.get("task_id", ""), tk_source)

    return {
        "type": "direct",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": tk_source},
        ],
    }


def make_comparison_example(entry: dict) -> dict | None:
    """Create a multi-language comparison example if references exist."""
    refs = entry.get("references", {})
    tk_source = entry["tk_source"]
    tk_tokens = entry.get("tk_tokens", 0)

    py_source = refs.get("python_source")
    py_tokens = refs.get("python_tokens", 0)

    if not py_source or not tk_tokens or not py_tokens:
        return None

    c_source = refs.get("c_source", "")
    c_tokens = refs.get("c_tokens", 0)

    ratio = py_tokens / tk_tokens if tk_tokens > 0 else 0

    prompt = (
        f"Show equivalent implementations in Python and toke, "
        f"comparing token efficiency.\n"
        f"[PYTHON {py_tokens}t]\n{py_source}"
    )
    if c_source:
        prompt += f"\n[C {c_tokens}t]\n{c_source}"

    response = (
        f"[TK {tk_tokens}t]\n{tk_source}\n"
        f"[RATIO] {ratio:.1f}x fewer tokens than Python."
    )

    return {
        "type": "comparison",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def format_chatml(messages: list[dict]) -> str:
    """Format messages into ChatML template for Qwen models."""
    parts = [
        "<|im_start|>system\n"
        "You are a toke programming language expert. "
        "Write correct, idiomatic toke code.<|im_end|>"
    ]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def load_corpus(path: Path) -> list[dict]:
    """Load corpus JSONL file."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: malformed JSON at line {lineno}", file=sys.stderr)
                continue
            if not entry.get("tk_source"):
                continue
            entries.append(entry)
    return entries


def process_corpus(entries: list[dict]) -> list[dict]:
    """Convert corpus entries into training examples."""
    examples: list[dict] = []

    for entry in entries:
        # Every entry gets a direct generation example.
        examples.append(make_direct_example(entry))

        # Entries with references get comparison examples.
        comparison = make_comparison_example(entry)
        if comparison:
            examples.append(comparison)

    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    """Write examples as JSONL with ChatML-formatted text field."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "text": format_chatml(ex["messages"]),
                "type": ex["type"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Path to corpus.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for training data")
    parser.add_argument("--split", type=float, default=0.95,
                        help="Train split ratio (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    # Load corpus.
    print(f"Loading corpus from {args.corpus}...")
    entries = load_corpus(args.corpus)
    print(f"  Loaded {len(entries)} entries")

    # Generate training examples.
    print("Generating training examples...")
    examples = process_corpus(entries)
    print(f"  Generated {len(examples)} examples")

    # Count by type.
    type_counts: dict[str, int] = {}
    for ex in examples:
        t = ex["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Shuffle and split.
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split_idx = int(len(examples) * args.split)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    print(f"  Split: {len(train_examples)} train, {len(eval_examples)} eval")

    # Write output.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "eval.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(eval_path, eval_examples)
    print(f"  Written to {train_path} and {eval_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
