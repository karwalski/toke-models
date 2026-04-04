# toke Tokenizer Design

**Status:** Planned — Phase 2 work, after Gate 1 is passed.

## Goal

A 32,768-token BPE vocabulary trained exclusively on validated toke source
code. Target: 2.5–4x token density improvement over cl100k_base.

## Approach

1. Extract all toke source from the validated Phase 1 corpus
2. Train BPE tokenizer with vocab size 32,768
3. Evaluate against cl100k_base on a held-out set of toke programs
4. Gate criterion: must achieve ≥2.5x improvement to proceed

## Dependencies

- Validated Phase 1 corpus (≥60,000 entries)
- Gate 1 passed
