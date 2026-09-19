# toke-model

toke-model fine-tunes open-source language models to write code in the [toke](https://github.com/karwalski/toke) programming language. It takes a corpus of example toke programs (produced by the toke-corpus pipeline), prepares it into chat-format training data, and trains and evaluates adapters on top of open base models.

**Status.** Every trained-model result on record is historical and v0.3-era. Gate 1 (2026-04-03): **58.8% functional Pass@1** (588 passed / 1,000 generated), Qwen 2.5 Coder 7B + QLoRA; the compile rate was 92.3% (923/1,000). *Corrected 2026-09-19 (story 128.19): this line previously read "63.7% compile Pass@1" and was wrong twice over.* The number was 588/**923** — the harness dropped the 77 generated solutions that failed to compile out of the denominator, which measures Pass@1 *given that the solution compiled*, not Pass@1; a solution that fails to compile is a failed attempt, not an absent one, so the denominator is the 1,000 solutions generated. And the label was wrong: the figure is *functional* Pass@1 (passes every hidden test), not a compile rate. **58.8% is below Gate 1's own declared `pass_at_1_minimum: 0.60`, so the Gate 1 verdict is re-opened and is the owner's to make** — see `huggingface/eval_results.json` and `toke-eval/docs/suspect-numbers-128-1c.md`. Gate 2 (2026-05-22): **100% compile Pass@1** and **55.6% functional** (272/489) — on the *curated* 500-hidden + 200-eval set the model was optimised against, and the 100% is never to be quoted without it. The honest floor is the full-local re-audit of all 1,748 v0.3.9 corpus programs: **37.5% compile** (655/1,748), **about 2.2% fully correct** (38 PASS). No v0.4-native model exists. Epic 128 — the training strategy reset (base-model bake-off, fine-tune vs from-scratch; see `toke/docs/progress.md`) — supersedes that pipeline choice. The next training run follows the Epic 128 plan against the Epic 129 corpus freeze, so treat the commands below as the mechanics of the previous runs, not a statement of the current strategy.

## About toke

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification.

toke is a compiled programming language designed for LLM code generation. It has 14
keywords, a 59-character set, a backtrack-free grammar with bounded lookahead, and one
canonical form per construct, chosen by measurement in a 46-pattern catalogue and
reproduced by `tkc --min`. That makes generated code cheap to constrain during decoding,
cheap for a compiler to verify afterwards, and compact to emit. Token efficiency is one
measured property of toke, always reported with its tokenizer and its baseline, not the
whole claim.

*The one-liner and the paragraph above are reproduced word for word from the canonical
description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

## What This Repository Contains

| Directory | Purpose |
|-----------|---------|
| `finetune/` | Fine-tuning scripts for MLX (Apple Silicon) and QLoRA (CUDA), with configs in `finetune/configs/` |
| `tokenizer/` | BPE tokenizer work for toke syntax (canonical tokenizer development lives in the toke-tokenizer repo) |
| `model/`, `train/` | Experimental from-scratch model definition and training loop |
| `scripts/` | Utilities for benchmark generation, adapter evaluation, and data conversion |
| `eval/` | Model safety evaluation via adversarial prompt testing |
| `benchmark/` | Benchmark task sets and validation reports |
| `ollama/` | Conversion + publishing of a merged model to the Ollama registry (`karwalski/toke`) |
| `docker/` | Self-hosted inference via HuggingFace TGI (GPU and CPU compose files) |
| `huggingface/` | Model card and upload assets for Hugging Face releases |
| `cloud/` | AWS training helper scripts |
| `docs/` | Training methodology, security evaluations, and pointers to relocated assets |
| `tests/` | Unit tests for the data preparation and model code |

There is no `corpus/` directory any more: the vendored corpus pipeline was archived in story 130.5. See [docs/corpus-location.md](docs/corpus-location.md) — the canonical, maintained pipeline is the **toke-corpus** repo.

## Where the Big Artifacts Live

Nothing large is tracked in git:

- `training-data/` — prepared train/eval JSONL, gitignored, generated locally from a corpus export.
- `output/` — merged model weights (e.g. `output/7b-merged/`), local only, never committed.
- `checkpoints/`, `logs/`, `results/` — run artifacts, local only.
- Gate-2-era runs, checkpoints, eval outputs, and the old vendored corpus are archived at `~/tk/archive/toke-model-gate2-era-20260819/` (with `MANIFEST.md`).
- Released weights go to Hugging Face when gate criteria are met; they are never stored in this repository.

## Related Repositories

| Repository | Relationship |
|------------|--------------|
| [toke](https://github.com/karwalski/toke) | The language: compiler (`tkc`) and standard library; also hosts `docs/progress.md`, the project plan of record |
| [toke-corpus](https://github.com/karwalski/toke-corpus) | Upstream: generates and quality-gates the training corpus this repo consumes |
| [toke-tokenizer](https://github.com/karwalski/toke-tokenizer) | Canonical custom-tokenizer training and packaging |
| [toke-eval](https://github.com/karwalski/toke-eval) | Downstream: benchmark tasks and the pass@k harness used to measure trained models |
| [toke-mcp](https://github.com/karwalski/toke-mcp) | MCP server for AI tool integration |

## Getting Started

### Requirements

- Python 3.10+
- Apple Silicon Mac with [MLX](https://github.com/ml-explore/mlx) or an NVIDIA GPU with CUDA
- The [toke compiler](https://github.com/karwalski/toke) (`tkc`) for validating generated code
- A corpus JSONL exported from [toke-corpus](https://github.com/karwalski/toke-corpus)
- The [toke-eval](https://github.com/karwalski/toke-eval) suite for measuring results

### Prepare Training Data

```bash
# Chat-format train/eval split from a toke-corpus export
python finetune/prepare_data.py --corpus /path/to/corpus.jsonl --output-dir training-data/
```

### Fine-tuning on Apple Silicon (MLX)

```bash
# Convert prepared data to MLX format
python finetune/prepare_mlx_data.py --input-dir training-data/ --output-dir training-data/mlx/

# Train (configs in finetune/configs/)
python finetune/train_mlx.py --config finetune/configs/7b_mlx.yaml

# Merge adapter weights into the base model
python finetune/merge_mlx.py --adapter /path/to/adapter --output output/7b-merged/
```

### Fine-tuning with QLoRA (CUDA GPU)

```bash
python finetune/train_qlora.py --config finetune/configs/7b.yaml
```

### Evaluating a Trained Model

Evaluation uses the [toke-eval](https://github.com/karwalski/toke-eval) benchmark:

```bash
python -m toke_eval.pass_at_k --solutions-dir solutions/ --tests-dir hidden_tests/ --compiler tkc
```

### Model Safety Testing

```bash
python eval/safety_eval.py --model-path /path/to/model \
    --llamaguard-path /path/to/llamaguard --output-dir /tmp/eval-out
```

Add `--dry-run` to exercise the harness without model weights. See [docs/security/model-safety-evals.md](docs/security/model-safety-evals.md) for methodology and results.

## Serving a Merged Model

- **Ollama:** `ollama/` converts a merged model to GGUF and publishes it as `karwalski/toke` — see [ollama/README.md](ollama/README.md).
- **Docker:** `docker/` runs self-hosted inference with HuggingFace TGI (GPU `docker-compose.yml`, CPU `docker-compose.cpu.yml`) — see [docker/README.md](docker/README.md).

## Licence

Apache 2.0. Model weights released under Apache 2.0 consistent with the Qwen 2.5 base model licence.
