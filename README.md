# toke-models

Fine-tuning scripts and evaluation harness for toke language models.

## Requirements

- Python 3.11+
- Apple Silicon Mac with MLX (for training)
- [toke-corpus](https://github.com/karwalski/toke-corpus) — validated corpus
- [toke-tokenizer](https://github.com/karwalski/toke-tokenizer) — Phase 2 tokenizer
- [toke-benchmark](https://github.com/karwalski/toke-benchmark) — evaluation tasks

## Fine-tuning

    # Prepare training data
    python finetune/prepare_data.py \
      --corpus /path/to/corpus \
      --out training-data/ \
      --format instruction

    # Fine-tune Qwen 2.5 Coder 7B
    python finetune/train_qlora.py --config finetune/configs/7b.yaml

## Evaluation

    python eval/run_benchmark.py \
      --model /path/to/model \
      --tasks /path/to/toke-benchmark/tasks/ \
      --out results/

## Model releases

Trained model weights are not stored in this repository.
Releases are published separately when gate criteria are met.

## Licence

Apache 2.0. Model weights released under Apache 2.0 consistent
with the Qwen 2.5 base model licence.
