---
language:
  - toke
license: apache-2.0
library_name: transformers
tags:
  - toke
  - code-generation
  - fine-tuned
  - qlora
  - dora
  - mlx
  - domain-specific-language
base_model: Qwen/Qwen2.5-Coder-7B
datasets:
  - karwalski/toke-corpus
metrics:
  - pass_at_1
  - token_reduction
model-index:
  - name: toke-coder-7b
    results:
      - task:
          type: text-generation
          name: Toke Code Generation
        dataset:
          type: karwalski/toke-benchmark
          name: toke-benchmark
          split: test
        metrics:
          - type: pass_at_1
            value: 63.7
            name: Pass@1
          - type: token_reduction
            value: 12.5
            name: Token Reduction (%)
---

# toke-coder-7b

A fine-tuned code generation model for the **toke** programming language, based on Qwen 2.5 Coder 7B.

## Model Description

toke-coder-7b is a QLoRA/DoRA adapter fine-tuned on top of [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) for generating syntactically correct and semantically valid toke source code. Toke is a domain-specific programming language designed for concise, structured program representation with significant token efficiency gains over general-purpose languages.

The model was trained on Apple Silicon hardware using MLX and achieves strong results on held-out toke programming tasks.

- **Base model:** Qwen 2.5 Coder 7B
- **Fine-tuning method:** QLoRA with DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Training framework:** MLX (Apple Silicon native)
- **Reward modelling:** GRPO (Group Relative Policy Optimisation)
- **Developed by:** [karwalski](https://github.com/karwalski)
- **License:** Apache 2.0

## Intended Uses

### Primary use

- Generating toke source code from natural language descriptions or partial code prompts.
- Code completion and infilling for toke programs.
- Assisting developers learning the toke language.

### Out-of-scope uses

- This model is trained exclusively on toke language data. It is not intended for general-purpose code generation in other languages.
- The model should not be used to generate code for safety-critical systems without human review.
- Not suitable for generating natural language prose or non-code content.

## Training Data

The model was fine-tuned on **toke-corpus**, a curated dataset of 46,000+ validated toke programs covering:

- Arithmetic and logical expressions
- Control flow (if/else, loops, match)
- Function definitions and calls
- Type declarations and struct types
- Module imports and namespacing
- Error handling patterns
- Standard library usage

All corpus entries are compilation-verified against the toke compiler (`tkc`). The corpus includes both Phase 1 (core language) and Phase 2 (advanced features) programs.

Source repository: [karwalski/toke-corpus](https://github.com/karwalski/toke-corpus)

## Training Procedure

### Hardware

- Apple Mac Studio, M4 Max
- Training performed entirely on Apple Silicon using MLX

### Hyperparameters

- **Method:** QLoRA with DoRA adapters
- **Epochs:** 1
- **Training loss:** 0.197
- **Eval loss:** 0.158
- **Training runtime:** ~23.6 hours
- **Precision:** Mixed (MLX native)

### Fine-tuning details

1. **Data preparation:** Corpus entries converted to instruction-following format (prompt/completion pairs).
2. **Adapter training:** QLoRA adapters trained on Qwen 2.5 Coder 7B base weights using MLX.
3. **Reward modelling:** GRPO reward model trained to prefer compilable, token-efficient toke output.
4. **Adapter merging:** Trained adapters merged into base model for inference.

## Evaluation Results

Evaluated on 1,000 held-out benchmark tasks from [toke-benchmark](https://github.com/karwalski/toke-benchmark).

| Metric | Value | Gate 1 Threshold |
|--------|-------|-------------------|
| Compilation rate | 92.3% (923/1000) | — |
| Pass@1 | **63.7%** (588/923) | >60% |
| Token reduction | **12.5%** (8K vocab) | >10% |
| Token reduction | 13.1% (32K vocab) | — |

**Gate 1 verdict:** PASS (2026-04-03)

### Benchmark methodology

- 500 original + 500 expanded held-out tasks
- Each task tested against hidden test inputs
- Compilation checked via `tkc` compiler
- Pass@1 measured as fraction of compilable solutions that produce correct output

## Ethical Considerations

- **Training data provenance:** All training data is synthetically generated and manually curated. No copyrighted code or personally identifiable information is included.
- **Bias:** The model is narrowly scoped to a single domain-specific language. It does not generate natural language and has limited potential for harmful text generation.
- **Dual use:** Toke is a research language. The model's capabilities are confined to toke code generation and are unlikely to enable harmful applications.
- **Environmental impact:** Training was performed on consumer Apple Silicon hardware with modest energy consumption (~24 hours on a single Mac Studio).

## Limitations

- The model only generates toke code. Prompts in other programming languages will produce poor results.
- Complex multi-module programs may require iterative generation and human review.
- The model was trained on Phase 1 and Phase 2 corpus data; language features added after the training cutoff are not supported.
- Generation quality degrades for programs significantly longer than those in the training distribution.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("karwalski/toke-coder-7b")
tokenizer = AutoTokenizer.from_pretrained("karwalski/toke-coder-7b")

prompt = "Write a toke function that returns the factorial of n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{toke-coder-7b,
  title={toke-coder-7b: Fine-tuned Code Generation for the Toke Language},
  author={karwalski},
  year={2026},
  url={https://huggingface.co/karwalski/toke-coder-7b}
}
```

## Model Card Contact

For questions or issues, open an issue on [GitHub](https://github.com/karwalski/toke-models).
