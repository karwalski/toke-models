#!/usr/bin/env python3
"""Prepare toke corpus for QLoRA fine-tuning.

Converts corpus.jsonl entries into instruction-tuning format per RFC Section 15.3.
Produces two example types:
  1. Direct generation — prompt -> toke code
  2. Multi-language comparison — side-by-side Python (and C) and toke source
     (no token counts: story 131.97)

(Story 128.26: an earlier docstring also promised "Correction" examples — broken
code + error -> fixed code. None were ever generated, and the corpus records this
script reads carry no broken-code or diagnostic fields to build them from, so the
promise is removed rather than implemented. Repair pairs are corpus-side work.)

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
from typing import Any

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


# Corpus records excluded from training prep by owner decision, matched on the
# record's "id". Story 135.34 / toke/docs/owner-decisions-2026-10-01.md §4.5: this
# legacy phase2 BIFI record carries tkc 0.1.0's retired W1010 advice-as-fix string
# in references.diagnostic. Keep in step with toke-corpus
# scripts/prepare_training_data.py EXCLUDED_RECORD_IDS.
EXCLUDED_RECORD_IDS: frozenset[str] = frozenset({
    "BIFI-missing_semicolon-21952-7371771c",
})

def _skip_string(src: str, i: int) -> int:
    """Return the index just past the string literal whose opening quote is at i."""
    j = i + 1
    while j < len(src):
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == '"':
            return j + 1
        j += 1
    return j


def extract_signature(tk_source: str) -> str:
    """Return the signature of the first top-level function definition, or "".

    A toke function definition is ``f=name(p1:T1;p2:T2):R{body};`` (``F=`` in
    legacy sources). ``;`` separates parameters as well as declarations, and a
    type may itself contain parentheses (``@(str:i64)``, ``(i64;i64):bool``),
    so the signature is read with a scanner rather than by splitting on ``;``:
    a declaration starts at depth 0 after the start of the source or a ``;``,
    its parameter list runs to the matching ``)``, and the return spec runs to
    the first ``{`` outside any bracket. The result is ``name(params):R``.
    Body-less declarations (``F=ext(s:Str):f64!Err;``, an extern) are skipped,
    so the prompt names the function the source actually defines.
    """
    n = len(tk_source)
    i = 0
    depth = 0  # {} nesting: only depth-0 text holds declarations
    at_decl_start = True
    while i < n:
        c = tk_source[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            i = _skip_string(tk_source, i)
            at_decl_start = False
            continue
        if depth == 0 and at_decl_start and tk_source.startswith(("f=", "F="), i):
            sig, end = _read_signature(tk_source, i + 2)
            if sig or end < 0:
                return sig
            i = end  # body-less declaration: resume at its terminating ';'
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(depth - 1, 0)
        at_decl_start = depth == 0 and c == ";"
        i += 1
    return ""


def _read_signature(src: str, start: int) -> tuple[str, int]:
    """Read ``name(params):R{`` from start.

    Returns ``(signature, -1)`` for a definition, ``("", index of ';')`` for a
    body-less declaration, and ``("", -1)`` if the text is not well formed.
    """
    paren = src.find("(", start)
    if paren <= start or not src[start:paren].strip().isidentifier():
        return "", -1
    depth = 0
    closed = False
    for i in range(paren, len(src)):
        c = src[i]
        if c == '"':
            return "", -1
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth == 0 and c == ")":
                closed = True
        elif depth == 0 and closed:
            if c == "{":
                return src[start:i].strip(), -1
            if c == ";":
                return "", i
            if c == "}":
                return "", -1
    return "", -1


def task_id_to_prompt(task_id: str, tk_source: str) -> str:
    """Generate a natural-language instruction from task metadata and source."""
    parts = task_id.split("-")
    cat = "-".join(parts[:2]) if len(parts) >= 2 else task_id
    cat_desc = CATEGORY_DESC.get(cat, "general programming")

    func_sig = extract_signature(tk_source)

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


def make_direct_example(entry: dict[str, Any]) -> dict[str, Any]:
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


def make_comparison_example(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Create a multi-language comparison example if references exist."""
    refs = entry.get("references", {})
    tk_source = entry["tk_source"]
    tk_tokens = entry.get("tk_tokens", 0)

    py_source = refs.get("python_source")
    py_tokens = refs.get("python_tokens", 0)

    if not py_source or not tk_tokens or not py_tokens:
        return None

    c_source = refs.get("c_source", "")

    # Story 132.15: this used to append "[RATIO] {x}x fewer tokens than Python" to every
    # comparison example, baking a withdrawn claim into the training data — and a
    # direction the measurements contradict (under one shared tokenizer toke costs
    # 1.34x [1.22, 1.48] the tokens of equivalent Python, N = 60; see
    # toke/docs/metrics-baseline.md).
    # Story 131.97: the per-side counts went too. They were written as
    # "[PYTHON Nt]" / "[C Nt]" / "[TK Nt]" with no tokenizer named, and nothing in
    # the corpus record says which tokenizer produced python_tokens, c_tokens or
    # tk_tokens, or that it was the same one for every side — two such counts are
    # the same unsourced ratio in two numbers (TEMSpec §6.3; owner decision §4.14).
    # The section markers stay, bare. Which records get a comparison example is
    # unchanged (the counts must still be present on the record).
    prompt = (
        f"Show equivalent implementations in Python and toke.\n"
        f"[PYTHON]\n{py_source}"
    )
    if c_source:
        prompt += f"\n[C]\n{c_source}"

    response = f"[TK]\n{tk_source}"

    return {
        "type": "comparison",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def format_chatml(messages: list[dict[str, Any]]) -> str:
    """Format messages into ChatML template for Qwen models."""
    parts = [
        (
            "<|im_start|>system\n"
            "You are a toke programming language expert. "
            "Write correct, idiomatic toke code.<|im_end|>"
        )
    ]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load corpus JSONL file."""
    entries = []
    excluded = 0
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
            if entry.get("id") in EXCLUDED_RECORD_IDS:
                excluded += 1
                continue
            entries.append(entry)
    if excluded:
        print(f"  Excluded {excluded} record(s) by owner decision (EXCLUDED_RECORD_IDS)")
    return entries


def process_corpus(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert corpus entries into training examples."""
    examples: list[dict[str, Any]] = []

    for entry in entries:
        # Every entry gets a direct generation example.
        examples.append(make_direct_example(entry))

        # Entries with references get comparison examples.
        comparison = make_comparison_example(entry)
        if comparison:
            examples.append(comparison)

    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
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
