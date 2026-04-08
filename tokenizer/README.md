# toke-model/tokenizer

Purpose-built BPE tokenizer for Phase 2 of
[toke](https://github.com/karwalski/toke).

## What this produces

A 32,768-token BPE vocabulary trained exclusively on validated toke
source code. Achieves 2.5–4x token density improvement over cl100k_base
for toke programs.

## Requirements

- Python 3.11+
- Validated Phase 1 corpus from [toke-model/corpus](../corpus/)

## Usage

    # Prepare corpus for tokenizer training
    python prepare.py --corpus /path/to/corpus --out tokenizer-data/

    # Train the tokenizer
    python train.py --data tokenizer-data/ --vocab-size 32768 --out tokenizer/

    # Evaluate against cl100k_base
    python eval.py --tokenizer tokenizer/ --benchmark /path/to/benchmark/tasks/

## Licence

Apache 2.0.
