# Hybrid MLX + CUDA Training Infrastructure

Story 10.7.9 — The toke training pipeline supports two frameworks that share the
same YAML config format and JSONL data pipeline, allowing seamless switching
between local Apple Silicon development and cloud NVIDIA GPU training.

## When to use each framework

### MLX (Apple Silicon)

Use `train_mlx.py` / `scripts/train_dora.sh` when:

- Running on Mac Studio M4 Max (128 GB unified memory)
- Small experiments: hyperparameter sweeps, adapter type comparisons
- Iterating on data pipeline changes (short feedback loop)
- Evaluating new corpus versions before committing to a full run
- Privacy-sensitive work that must stay on local hardware

Typical hardware: Mac Studio M4 Max, 128 GB unified memory, ~40 GB available
for model + training. Handles 7B models in 4-bit quantization comfortably.

### CUDA (Cloud GPU)

Use `scripts/train_cuda.sh` / `cloud/aws_train.sh` when:

- Running full training campaigns (all epochs, full corpus)
- Training larger models (32B+) that exceed unified memory
- Multi-GPU distributed training is needed
- Reproducible benchmark runs for paper/RFC results
- Longer sequences (4096+) that require more VRAM

Typical hardware: AWS g5.xlarge (A10G 24 GB), g5.2xlarge for 32B models,
or p4d instances for multi-GPU. Cost: ~$0.35-0.50/hr spot for g5.xlarge.

## Config format compatibility

Both frameworks read the same YAML config structure. The shared sections are:

```yaml
model:
  base: "Qwen/Qwen2.5-Coder-7B-Instruct"
  quantization: true          # MLX: native 4-bit; CUDA: mapped to nf4

lora:
  rank: 64
  alpha: 128.0
  dropout: 0.05
  use_dora: true              # Both frameworks support DoRA

training:
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 100
  max_seq_length: 2048

data:
  train_file: "training-data/train.jsonl"
  eval_file: "training-data/eval.jsonl"

output:
  adapter_dir: "output/7b-dora/adapter"
```

### Framework-specific keys

Some config keys are framework-specific but safely ignored by the other:

| Key | MLX | CUDA | Notes |
|-----|-----|------|-------|
| `lora.keys` | Target layer names | Mapped to `target_modules` | MLX-style layer names work in both |
| `lora.target_modules` | Ignored | PEFT target modules | Use `keys` for cross-framework compat |
| `lora.task_type` | Ignored | PEFT task type | Defaults to `CAUSAL_LM` |
| `lora.bias` | Ignored | PEFT bias mode | Defaults to `none` |
| `model.bnb_4bit_compute_dtype` | Ignored | BitsAndBytes dtype | Defaults to `bfloat16` |
| `model.bnb_4bit_use_double_quant` | Ignored | Double quantization | Defaults to `true` |
| `training.train_embeddings` | Unfreezes embed/lm_head | Unfreezes embed/lm_head | Same behavior |
| `training.grad_checkpoint` | MLX gradient checkpointing | Mapped to `gradient_checkpointing` | Both supported |
| `training.optim` | Ignored | HF optimizer string | Defaults to `paged_adamw_8bit` |
| `training.bf16` | Ignored (MLX uses native) | Enables bf16 training | Defaults to `true` |

### Best practice

Write configs using MLX-style keys (`keys`, `grad_checkpoint`, `save_every`,
`steps_per_report`, `steps_per_eval`). The CUDA script maps these automatically.
This way a single config file works with both `train_mlx.py` and
`scripts/train_cuda.sh`.

## Data pipeline

Both frameworks consume the same JSONL corpus format:

```json
{"text": "<|im_start|>system\nYou are a toke programming assistant...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
```

The training data lives in `training-data/train.jsonl` and
`training-data/eval.jsonl`. Both are generated from the toke corpus by
`finetune/prepare_data.py`.

No conversion is needed when switching frameworks.

## Checkpoint transfer between frameworks

MLX and CUDA use different weight formats, but adapters can be transferred:

### MLX to CUDA

MLX saves adapters as `adapters.safetensors` (safetensors format). To load
these in a CUDA/PEFT environment:

```python
from safetensors.torch import load_file

# Load MLX adapter weights.
mlx_weights = load_file("output/7b-mlx-dora/adapter/adapters.safetensors")

# MLX and PEFT use different key naming conventions.
# Map MLX keys (e.g., "model.layers.0.self_attn.q_proj.lora_a")
# to PEFT keys (e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight").
# The key differences:
#   - PEFT prefixes with "base_model.model."
#   - PEFT uses "lora_A.weight" / "lora_B.weight" (capitalised, with .weight suffix)
#   - DoRA magnitude vectors: MLX "magnitude" -> PEFT "lora_magnitude_vector"

def mlx_key_to_peft(key: str) -> str:
    key = "base_model.model." + key
    key = key.replace(".lora_a", ".lora_A.weight")
    key = key.replace(".lora_b", ".lora_B.weight")
    key = key.replace(".magnitude", ".lora_magnitude_vector")
    return key

peft_weights = {mlx_key_to_peft(k): v for k, v in mlx_weights.items()}
```

### CUDA to MLX

PEFT saves adapters via `model.save_pretrained()` producing
`adapter_model.safetensors` and `adapter_config.json`. To convert:

```python
from safetensors.torch import load_file
import mlx.core as mx
import numpy as np

peft_weights = load_file("output/cuda/adapter/adapter_model.safetensors")

def peft_key_to_mlx(key: str) -> str:
    key = key.replace("base_model.model.", "")
    key = key.replace(".lora_A.weight", ".lora_a")
    key = key.replace(".lora_B.weight", ".lora_b")
    key = key.replace(".lora_magnitude_vector", ".magnitude")
    return key

mlx_weights = {}
for k, v in peft_weights.items():
    mlx_key = peft_key_to_mlx(k)
    mlx_weights[mlx_key] = mx.array(v.numpy())
```

### Important notes on transfer

- Quantization is not transferred. Each framework re-quantizes the base model
  at load time. Only the low-rank adapter weights (and DoRA magnitude vectors)
  are portable.
- The base model must be the same on both sides. Transferring adapters trained
  on `Qwen2.5-Coder-7B-Instruct` to a different base model will not work.
- After transfer, always run a validation pass on a few examples to confirm
  the adapter produces sensible output.

## Recommended hardware

| Use case | Framework | Hardware | Est. cost | Training time (7B, 73K examples) |
|----------|-----------|----------|-----------|----------------------------------|
| Local iteration | MLX | Mac Studio M4 Max 128 GB | $0 (owned) | ~8-12 hours |
| Cloud single-GPU | CUDA | AWS g5.xlarge (A10G 24 GB) | ~$3-5 spot | ~6-10 hours |
| Cloud fast | CUDA | AWS g5.2xlarge (A10G 24 GB, more CPU/RAM) | ~$5-8 spot | ~5-8 hours |
| Large model (32B) | CUDA | AWS g5.4xlarge or p4d | ~$15-30 spot | ~20-40 hours |
| Multi-GPU | CUDA | AWS p4d.24xlarge (8x A100) | ~$100+ spot | ~2-4 hours |

### GPU memory requirements (4-bit quantized)

| Model size | Min VRAM | Recommended |
|-----------|----------|-------------|
| 7B | 8 GB | 16+ GB |
| 14B | 14 GB | 24+ GB |
| 32B | 22 GB | 40+ GB |

## Quick start

### MLX (local Mac)

```bash
# DoRA training on Mac Studio
./scripts/train_dora.sh
```

### CUDA (cloud or local NVIDIA)

```bash
# Validate config without training
./scripts/train_cuda.sh --config finetune/configs/7b_mlx_dora.yaml --dry-run

# Full training run
./scripts/train_cuda.sh --config finetune/configs/7b_mlx_dora.yaml

# With custom output directory
./scripts/train_cuda.sh --config finetune/configs/7b_mlx_dora.yaml --output-dir output/cuda-dora

# Resume from checkpoint
./scripts/train_cuda.sh --config finetune/configs/7b_mlx_dora.yaml --resume output/cuda-dora/checkpoint-500
```

### Fire-and-forget cloud training

```bash
# Provisions AWS spot instance, trains, uploads results, self-terminates
export GITHUB_TOKEN=ghp_...
./cloud/aws_train.sh
```
