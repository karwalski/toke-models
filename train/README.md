# toke 1B Training

Training scaffold for the toke 1B purpose-built model.

## Setup

Requires Python 3.10+.

```bash
pip install mlx tokenizers
```

## Directory Layout

```
train/
  config.py        # Training hyperparameters (dataclass)
  train_1b.py      # Main training script (stub)
tokenizer/
  train_bpe.py     # BPE tokenizer training (stub)
  vocab/           # Trained tokenizer output
corpus/
  scripts/         # Corpus processing scripts
checkpoints/       # Model checkpoints (gitignored)
logs/              # Training logs as JSONL (gitignored)
```

## Usage

Train the BPE tokenizer first, then run the training script:

```bash
# 1. Train tokenizer
python -m tokenizer.train_bpe --corpus corpus/filtered/filtered.jsonl

# 2. Train model
python -m train.train_1b

# Resume from checkpoint
python -m train.train_1b --resume checkpoints/step_010000.safetensors
```

## Configuration

All hyperparameters live in `train/config.py` as a `TrainConfig` dataclass.
Import and override as needed:

```python
from train.config import TrainConfig

cfg = TrainConfig(total_steps=100_000, batch_size=64)
```

## Status

All scripts are stubs with TODO comments marking where implementation goes.
Blocked on local compute availability (Gate 2).
