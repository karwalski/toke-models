#!/usr/bin/env python3
"""Generate benchmark tasks from the toke corpus for Pass@1 evaluation.

Reads corpus_default.jsonl, filters to records with differential-tested
expected outputs, excludes records used in training, applies Phase 2
syntax autofix to all toke source, and writes benchmark tasks in the
format expected by eval_pass1_cuda.py.

Every tk_source and signature hint in the output is guaranteed to be
Phase 2 clean: lowercase identifiers, no underscores, semicolons as
separators, no square brackets, no Python code in prompts.

Each benchmark task contains:
  - task_id: original corpus record ID
  - description: natural-language task description derived from the record
  - expected_output: majority_output from differential testing
  - reference_source: the known-good toke source (for debugging, not shown to model)

Usage:
    python scripts/generate_benchmark_tasks.py \
        --corpus /path/to/toke-corpus/data/corpus_default.jsonl \
        --training-data /path/to/toke-corpus/data/train.jsonl \
        --output benchmark/tasks.jsonl \
        --max-tasks 400
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import Phase 2 autofix from the corpus pipeline
# ---------------------------------------------------------------------------
CORPUS_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "toke-corpus" / "scripts"
sys.path.insert(0, str(CORPUS_SCRIPTS))
from phase2_syntax_audit import autofix_source, detect_violations  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_training_sources(training_path: Path) -> set[str]:
    """Extract toke source strings from training data to identify used records.

    Training data is in messages format or ChatML text format.
    """
    sources = set()
    with open(training_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Messages format
            messages = record.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    sources.add(msg["content"].strip())
            # ChatML text format
            text = record.get("text", "")
            if "<|im_start|>assistant" in text:
                parts = text.split("<|im_start|>assistant")
                if len(parts) >= 2:
                    code = parts[-1].replace("<|im_end|>", "").strip()
                    sources.add(code)
    return sources


# ---------------------------------------------------------------------------
# Phase 2 validation
# ---------------------------------------------------------------------------

# Characters allowed outside strings in Phase 2 (56-char set).
_ALLOWED_OUTSIDE_STRINGS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "(){}=:.;+-*/<>!|$@\"\\ \t\n\r"
)

_CAMEL_RE = re.compile(r"[a-z]+[A-Z]")
_UPPER_TYPE_RE = re.compile(r":([A-Z][a-z]+)")  # :Str, :Bool etc.


def is_phase2_clean(src: str) -> bool:
    """Return True if src has no Phase 2 violations."""
    violations = detect_violations(src)
    return len(violations) == 0


def phase2_clean_source(src: str) -> str | None:
    """Apply Phase 2 autofix and return cleaned source, or None if unfixable."""
    if is_phase2_clean(src):
        return src
    fixed, collisions, fixable = autofix_source(src)
    if not fixable:
        return None
    # Verify the fix actually worked
    if not is_phase2_clean(fixed):
        return None
    return fixed


# ---------------------------------------------------------------------------
# Signature and description generation
# ---------------------------------------------------------------------------

def _toke_signature_hint(tk_source: str) -> str:
    """Extract toke function signatures (excluding main) from source."""
    hints = []
    for part in tk_source.split("f="):
        if not part or part.startswith("main("):
            continue
        brace = part.find("{")
        if brace == -1:
            continue
        sig = "f=" + part[:brace].rstrip()
        if "(" in sig and ")" in sig:
            hints.append(sig)
    return "; ".join(hints) if hints else ""


def derive_description(record: dict) -> str:
    """Derive a natural-language task description from a corpus record.

    Uses only toke syntax and natural language — no Python/C/Java code.
    """
    task_id = record.get("task_id", "")
    tk_source = record.get("tk_source", "")

    # Module name
    mod_name = ""
    if tk_source.startswith("m="):
        mod_name = tk_source.split(";")[0].replace("m=", "")

    # Toke signature hint (already Phase 2 cleaned at this point)
    toke_hint = _toke_signature_hint(tk_source) if tk_source else ""

    # Extract docstring from Python reference (prose only, no code).
    refs = record.get("references", {})
    py_src = refs.get("python_source", "")
    py_doc = ""
    if py_src:
        lines = py_src.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("def ") and "main" not in line:
                for j in range(i + 1, min(i + 10, len(lines))):
                    stripped = lines[j].strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        doc = stripped.strip("\"'").strip()
                        if doc:
                            py_doc = doc
                        break
                break

    # Category from task_id
    category_map = {
        "ARR": "array operations",
        "STR": "string manipulation",
        "MATH": "mathematical computation",
        "MTH": "mathematical computation",
        "SRT": "sorting",
        "SORT": "sorting",
        "SEARCH": "searching",
        "TREE": "tree operations",
        "GRAPH": "graph algorithms",
        "DP": "dynamic programming",
        "HASH": "hash map operations",
        "IO": "input/output",
        "BIT": "bitwise operations",
        "REC": "recursion",
        "STACK": "stack operations",
        "QUEUE": "queue operations",
        "LINK": "linked list operations",
        "GEO": "geometry",
        "SIM": "simulation",
        "PARSE": "parsing",
        "CONV": "type conversion",
        "CND": "conditional logic",
        "BOOL": "boolean logic",
        "CTRL": "control flow",
        "ITER": "iteration patterns",
        "ERR": "error handling",
        "FILE": "file operations",
        "HTTP": "HTTP operations",
        "JSON": "JSON processing",
        "FMT": "formatting",
        "AGG": "aggregation",
        "FILT": "filtering",
        "MAP": "mapping/transformation",
        "RED": "reduction",
    }

    parts = task_id.split("-") if task_id else []
    category_code = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 2 else "")
    category_desc = category_map.get(category_code, "general programming")

    # Build description — no non-toke code anywhere.
    desc_parts = [f"Write a toke program for {category_desc}."]
    if mod_name:
        desc_parts.append(f"The module should be named '{mod_name}'.")
    if toke_hint:
        desc_parts.append(f"Signature: {toke_hint}")
    if py_doc:
        desc_parts.append(py_doc)

    # Expected output hint
    majority_output = record.get("differential", {}).get("majority_output", "")
    if majority_output:
        output_lines = majority_output.strip().split("\n")
        if len(output_lines) <= 5:
            desc_parts.append(
                "The program should print the following output (one value per line):\n"
                + "\n".join(output_lines)
            )
        else:
            desc_parts.append(
                f"The program should print {len(output_lines)} lines of output, "
                f"starting with: {output_lines[0]}"
            )

    return "\n".join(desc_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="Path to corpus_default.jsonl")
    parser.add_argument("--training-data", type=Path, default=None,
                        help="Path to train.jsonl (messages format) to exclude training records")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output benchmark tasks JSONL file")
    parser.add_argument("--max-tasks", type=int, default=400,
                        help="Maximum number of benchmark tasks to generate")
    parser.add_argument("--min-languages", type=int, default=2,
                        help="Minimum number of languages that agreed on output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load corpus.
    print(f"Loading corpus: {args.corpus}")
    corpus = load_jsonl(args.corpus)
    print(f"  Total records: {len(corpus):,}")

    # Load training sources to exclude (check both raw and autofixed forms).
    training_sources: set[str] = set()
    if args.training_data and args.training_data.exists():
        print(f"Loading training data for exclusion: {args.training_data}")
        training_sources = extract_training_sources(args.training_data)
        print(f"  Training sources: {len(training_sources):,}")

    # Filter and clean eligible benchmark records.
    eligible = []
    skipped_no_diff = 0
    skipped_no_agree = 0
    skipped_no_judge = 0
    skipped_no_source = 0
    skipped_in_training = 0
    skipped_unfixable = 0
    skipped_dedup = 0
    seen_sources: set[str] = set()

    for rec in corpus:
        # Must have differential output.
        diff = rec.get("differential", {})
        majority_output = diff.get("majority_output", "")
        if not majority_output or not majority_output.strip():
            skipped_no_diff += 1
            continue

        # Must have enough language agreement.
        languages = diff.get("languages_agreed", [])
        if len(languages) < args.min_languages:
            skipped_no_agree += 1
            continue

        # Must be accepted by judge.
        judge = rec.get("judge", {})
        if not judge.get("accepted", False):
            skipped_no_judge += 1
            continue

        # Must have toke source.
        tk_source = rec.get("tk_source", "")
        if not tk_source:
            skipped_no_source += 1
            continue

        # Apply Phase 2 autofix to the toke source.
        cleaned = phase2_clean_source(tk_source)
        if cleaned is None:
            skipped_unfixable += 1
            continue

        # Exclude if in training set (check both original and cleaned forms).
        if tk_source.strip() in training_sources or cleaned.strip() in training_sources:
            skipped_in_training += 1
            continue

        # Deduplicate by cleaned source.
        if cleaned in seen_sources:
            skipped_dedup += 1
            continue
        seen_sources.add(cleaned)

        # Store cleaned source back for description generation.
        rec["tk_source"] = cleaned
        eligible.append(rec)

    print(f"\n  Filtering results:")
    print(f"    No differential output:  {skipped_no_diff:,}")
    print(f"    Insufficient agreement:  {skipped_no_agree:,}")
    print(f"    Not judge-accepted:      {skipped_no_judge:,}")
    print(f"    No toke source:          {skipped_no_source:,}")
    print(f"    Phase 2 unfixable:       {skipped_unfixable:,}")
    print(f"    In training set:         {skipped_in_training:,}")
    print(f"    Duplicate (post-clean):  {skipped_dedup:,}")
    print(f"    Eligible for benchmark:  {len(eligible):,}")

    if len(eligible) == 0:
        print("ERROR: no eligible benchmark records found", file=sys.stderr)
        return 1

    # Sample benchmark tasks.
    n = min(args.max_tasks, len(eligible))
    sampled = rng.sample(eligible, n)
    print(f"  Sampled: {n}")

    # Generate benchmark tasks.
    tasks = []
    for rec in sampled:
        diff = rec.get("differential", {})
        task = {
            "task_id": rec.get("id", rec.get("task_id", "")),
            "description": derive_description(rec),
            "expected_output": diff["majority_output"].rstrip(),
            "reference_source": rec["tk_source"],
            "languages_agreed": diff.get("languages_agreed", []),
            "judge_score": rec.get("judge", {}).get("score", 0),
        }
        tasks.append(task)

    # Final validation: assert every task is Phase 2 clean.
    dirty = 0
    for task in tasks:
        ref = task["reference_source"]
        if not is_phase2_clean(ref):
            dirty += 1
            print(f"  WARNING: task {task['task_id']} reference_source still dirty!",
                  file=sys.stderr)
        desc = task["description"]
        # Check description for Python-style contamination.
        if "def " in desc and "(" in desc:
            print(f"  WARNING: task {task['task_id']} description contains Python code!",
                  file=sys.stderr)

    if dirty:
        print(f"\n  ERROR: {dirty} tasks have dirty reference sources", file=sys.stderr)
        return 1

    print(f"\n  Phase 2 validation: all {len(tasks)} tasks clean")

    # Write output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"\n  Written: {args.output}")
    print(f"  Tasks: {len(tasks)}")

    # Report category distribution.
    categories: dict[str, int] = {}
    for t in tasks:
        tid = t["task_id"]
        parts = tid.split("-") if tid else ["UNK"]
        cat = parts[2] if len(parts) >= 4 else (parts[1] if len(parts) >= 2 else "UNK")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Category distribution:")
    for cat in sorted(categories, key=categories.get, reverse=True):
        print(f"    {cat:12s}  {categories[cat]:4d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
