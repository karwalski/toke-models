#!/usr/bin/env python3
"""Convert OpenAI messages-format JSONL to ChatML text-format JSONL for MLX-LM.

Reads train.jsonl / eval.jsonl with {"messages": [...]} format and writes
JSONL with {"text": "<|im_start|>system\n...<|im_end|>\n..."} format.

Usage:
    python scripts/convert_messages_to_chatml.py \
        --input-dir /path/to/toke-corpus/data \
        --output-dir training-data-p2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def messages_to_chatml(messages: list[dict]) -> str:
    """Convert a list of chat messages to ChatML format string."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def convert_file(input_path: Path, output_path: Path) -> int:
    """Convert one JSONL file. Returns number of records converted."""
    count = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not messages:
                print(f"WARNING: record missing 'messages' field, skipping", file=sys.stderr)
                continue
            text = messages_to_chatml(messages)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing train.jsonl and eval.jsonl (messages format)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write ChatML-format train.jsonl and eval.jsonl")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.jsonl", "eval.jsonl"):
        inp = args.input_dir / name
        out = args.output_dir / name
        if not inp.exists():
            print(f"WARNING: {inp} not found, skipping", file=sys.stderr)
            continue
        n = convert_file(inp, out)
        print(f"  {name}: {n:,} records converted → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
