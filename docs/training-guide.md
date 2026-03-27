# toke Model Training Guide

**Status:** Planned — Phase 1 work after corpus generation is complete.

## Approach

QLoRA fine-tuning of Qwen 2.5 Coder 7B and 32B using MLX on Apple Silicon.

## Prerequisites

- Validated corpus from toke-corpus
- tkc binary on PATH for evaluation
- toke-benchmark task set

## Gate 1 Criteria

- >10% token reduction vs Python baseline
- Pass@1 ≥ 60% on held-out benchmark tasks
