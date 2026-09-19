<!--
  HISTORICAL — superseded, never published (story 132.4).

  This card described "toke-coder-7b", a model that was never uploaded: the only
  published model is https://huggingface.co/karwalski/toke (toke-7b-gate2), whose
  card is README.md beside this file. Every number below is Gate-1 era (April
  2026), measured on a corpus and a tokenizer that have both been superseded. The
  "12.5% token reduction" row compared two tokenizers on one text (8K toke BPE vs cl100k_base, N = 46,754 programs)
  and is withdrawn as a headline claim (story 132.6). It is retained as
  a record of what was believed at the time. Do not copy anything from it to a
  public surface; the live facts are in toke/docs/about/canonical.md and
  toke/docs/metrics-baseline.md.

  CORRECTED 2026-09-19 (story 128.19). The Pass@1 this card carried, 63.7%, was
  588/923 — it dropped the 77 generated solutions that failed to compile out of
  the denominator. The correct figure is 588/1000 = 58.8%. Both the model-index
  metric and the results table below have been corrected, with the original
  claim kept visible beside each. 58.8% is below Gate 1's declared
  pass_at_1_minimum of 0.60, so the Gate 1 verdict is re-opened; re-deciding it
  is an owner action and is not done here. Nothing in this file has been or is
  to be uploaded.
-->

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
  - karwalski/toke-models
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
          type: karwalski/toke-eval
          name: toke-eval/benchmark
          split: test
        metrics:
          # Corrected 2026-09-19 (story 128.19): 588 passed / 1000 generated.
          # Previously carried 63.7, which was 588/923 (non-compiling solutions
          # dropped from the denominator). Never uploaded.
          - type: pass_at_1
            value: 58.8
            name: Pass@1
            verified: false
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

The model was fine-tuned on **toke-model/corpus**, a curated dataset of 46,000+ validated toke programs covering:

- Arithmetic and logical expressions
- Control flow (if/else, loops, match)
- Function definitions and calls
- Type declarations and struct types
- Module imports and namespacing
- Error handling patterns
- Standard library usage

All corpus entries are compilation-verified against the toke compiler (`tkc`). The corpus includes both Phase 1 (core language) and Phase 2 (advanced features) programs.

Source repository: [karwalski/toke-models](https://github.com/karwalski/toke-models)

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

Evaluated on 1,000 held-out benchmark tasks from [toke-eval/benchmark](https://github.com/karwalski/toke-eval).

| Metric | Value | Gate 1 Threshold |
|--------|-------|-------------------|
| Compilation rate | 92.3% (923/1000) | — |
| Pass@1 (functional) | **58.8%** (588/1000) | >60% — **not met** |
| Token reduction | **12.5%** (8K vocab) | >10% |
| Token reduction | 13.1% (32K vocab) | — |

**Pass@1 correction (2026-09-19, story 128.19).** This row previously read
**63.7% (588/923)**. The derivation:

| Quantity | Value |
|---|---|
| Solutions generated | 1,000 (`benchmark/solutions/*.toke`) |
| Solutions that compiled | 923 |
| Solutions that passed every hidden test | 588 |
| Previously published "Pass@1" = 588/923 | 63.7% — **withdrawn** |
| **Corrected Pass@1 = 588/1000** | **58.8%** |

923 was the wrong denominator: `load_toke_solutions()` dropped every generated
solution that failed to compile, so those 77 never became a scored result and
left the denominator entirely. 588/923 is Pass@1 *given that the solution
compiled* — a different and strictly more generous quantity. A solution that
fails to compile is a failed attempt, not an absent one, so the denominator is
the 1,000 solutions generated. The error was visible inside `eval_results.json`,
which recorded `benchmark_size: 1000` alongside `pass_at_1: 0.637`. No
re-evaluation was run: the correction is arithmetic over artefacts already on
disk.

**Gate 1 verdict:** **OPEN** — originally recorded as PASS (2026-04-03) on the
withdrawn 63.7% figure. 58.8% is below the declared `pass_at_1_minimum: 0.60`,
so on its own criterion Gate 1 is not met. Re-deciding the verdict is the
owner's call and is not made here.

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
