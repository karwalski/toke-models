# toke-model

Fine-tuning scripts for toke language models.

## Status

Gate 1 passed (2026-04-03). Qwen 2.5 Coder 7B + QLoRA adapter achieved 63.7% Pass@1 on 1,000 held-out tasks.

## Requirements

- Python 3.10+
- Apple Silicon Mac with MLX (for local training)
- [toke-model/corpus](corpus/) — validated corpus (this repo)
- [toke-model/tokenizer](tokenizer/) — BPE tokenizer (this repo)
- [toke-eval/benchmark](https://github.com/karwalski/toke-eval) — evaluation tasks
- [toke-eval](https://github.com/karwalski/toke-eval) — evaluation toolkit

## Fine-tuning

### MLX (Apple Silicon — recommended)

    python finetune/prepare_mlx_data.py --corpus /path/to/corpus_p2.jsonl --out training-data/
    python finetune/train_mlx.py --config finetune/configs/7b_mlx.yaml
    python finetune/merge_mlx.py --adapter results/adapter-mlx --out merged-model/

### QLoRA (GPU)

    python finetune/prepare_data.py --corpus /path/to/corpus --out training-data/ --format instruction
    python finetune/train_qlora.py --config finetune/configs/7b.yaml

## Evaluation

Evaluation tools are in [toke-eval](https://github.com/karwalski/toke-eval):

    python -m toke_eval.pass_at_k --solutions-dir solutions/ --tests-dir hidden_tests/ --compiler tkc

## Safety

Model safety evaluation via adversarial prompt testing:

    python eval/safety_eval.py --model-path /path/to/model --output-dir /tmp/eval-out

See [docs/security/model-safety-evals.md](docs/security/model-safety-evals.md).

## Model releases

Trained model weights are not stored in this repository.
Releases are published to Hugging Face when gate criteria are met.

## Licence

Apache 2.0. Model weights released under Apache 2.0 consistent
with the Qwen 2.5 base model licence.
