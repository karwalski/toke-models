# toke-model

toke-model trains AI models to write code in the [toke](https://github.com/karwalski/toke) programming language. It takes a large collection of example toke programs, processes them into training data, and fine-tunes open-source language models so they can generate correct toke code from natural-language descriptions.

The current best result: a 7-billion-parameter model (Qwen 2.5 Coder 7B with QLoRA) that writes working toke programs 63.7% of the time on its first attempt, achieving Gate 1 certification (2026-04-03).

## What This Repository Contains

| Directory | Purpose |
|-----------|---------|
| `corpus/` | Training corpus — 188,830 deduplicated toke programs across 61 categories, with validation pipeline, quality scoring, and prompt templates |
| `tokenizer/` | Custom BPE tokenizer trained on toke syntax for efficient token usage |
| `finetune/` | Fine-tuning scripts for MLX (Apple Silicon) and QLoRA (CUDA GPU), with configuration files |
| `scripts/` | Utility scripts for benchmark generation, evaluation, and data conversion |
| `eval/` | Model safety evaluation via adversarial prompt testing |
| `training-data/` | Prepared training data in chat format with category-specific prompts |
| `docs/` | Training methodology, security evaluations, and corpus documentation |

## Getting Started

### Requirements

- Python 3.10+
- Apple Silicon Mac with [MLX](https://github.com/ml-explore/mlx) (recommended) or NVIDIA GPU with CUDA
- The [toke compiler](https://github.com/karwalski/toke) (`tkc`) for corpus validation
- The [toke-eval](https://github.com/karwalski/toke-eval) benchmark suite for measuring results

### Fine-tuning on Apple Silicon (MLX)

```bash
# Prepare training data from the corpus
python finetune/prepare_mlx_data.py --corpus corpus/corpus_p2.jsonl --out training-data/

# Train (uses configs in finetune/configs/)
python finetune/train_mlx.py --config finetune/configs/7b_mlx.yaml

# Merge adapter weights into base model
python finetune/merge_mlx.py --adapter results/adapter-mlx --out merged-model/
```

### Fine-tuning with QLoRA (CUDA GPU)

```bash
python finetune/prepare_data.py --corpus corpus/corpus_p2.jsonl --out training-data/ --format instruction
python finetune/train_qlora.py --config finetune/configs/7b.yaml
```

### Evaluating a Trained Model

Evaluation uses the [toke-eval](https://github.com/karwalski/toke-eval) benchmark:

```bash
python -m toke_eval.pass_at_k --solutions-dir solutions/ --tests-dir hidden_tests/ --compiler tkc
```

### Model Safety Testing

```bash
python eval/safety_eval.py --model-path /path/to/model --output-dir /tmp/eval-out
```

See [docs/security/model-safety-evals.md](docs/security/model-safety-evals.md) for methodology and results.

## Related Repositories

| Repository | Description |
|------------|-------------|
| [toke](https://github.com/karwalski/toke) | The toke language compiler and standard library |
| [toke-eval](https://github.com/karwalski/toke-eval) | Benchmark tasks and evaluation harness |
| [toke-mcp](https://github.com/karwalski/toke-mcp) | MCP server for AI tool integration |
| [toke-ooke](https://github.com/karwalski/toke-ooke) | Web framework written in toke |
| [toke-website](https://github.com/karwalski/toke-website) | tokelang.dev documentation site |

## Model Releases

Trained model weights are not stored in this repository. Releases will be published to Hugging Face when gate criteria are met.

## Licence

Apache 2.0. Model weights released under Apache 2.0 consistent with the Qwen 2.5 base model licence.
