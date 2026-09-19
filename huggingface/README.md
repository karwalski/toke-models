---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - toke
  - code-generation
  - programming-language
  - qwen2
  - qlora
  - fine-tuned
  - awq
  - 4bit
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
pipeline_tag: text-generation
model-index:
  - name: toke-7b-gate2
    results:
      - task:
          type: text-generation
          name: Code Generation (toke v0.3 syntax)
        metrics:
          - name: Compilation Pass@1 (curated 500-hidden + 200-eval set, 2026-05-22)
            type: pass@1
            value: 100
          - name: Functional Pass@1 (same curated set, 272/489)
            type: pass@1
            value: 55.6
          - name: Compilation Pass@1 (full-local re-audit, all 1,748 v0.3.9 corpus programs)
            type: pass@1
            value: 37.5
---

<!--
  This file is the model card for the PUBLISHED Hugging Face model
  https://huggingface.co/karwalski/toke (toke-7b-gate2). Story 132.4 rewrote it:
  the live card carried "52% fewer tokens", "13 keywords", a "55-character
  alphabet" and an unqualified "100% of the time", all of which are withdrawn or
  wrong (toke/docs/about/canonical.md §11, toke/docs/metrics-baseline.md).

  Every fact here comes from toke/docs/about/canonical.{md,json} and
  toke/docs/metrics-baseline.md. Change those first, then re-copy here, then run
  `make check-canonical` in the toke repo.

  Uploading this card to the Hub is an owner action — see UPLOAD.md and
  toke/docs/about/registry-descriptions.md.
-->

# toke-7b-gate2

A 7B code-generation model fine-tuned to write **toke**, published as
[`karwalski/toke`](https://huggingface.co/karwalski/toke).

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

- Website: [tokelang.dev](https://tokelang.dev)
- Compiler, specification and standard library:
  [github.com/karwalski/toke](https://github.com/karwalski/toke)
- Training and evaluation code:
  [github.com/karwalski/toke-models](https://github.com/karwalski/toke-models)

*The one-liner and the paragraph above are reproduced word for word from the canonical
description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

## Read this before quoting the model

**This model writes v0.3 toke, and the language is on v0.4.** No v0.4-native model
exists, and no model has been trained since April 2026. Output from this model will not
match the v0.4 specification, the v0.4 canonical forms, or `tkc --min` on a current
compiler; it is published as the Gate 2 research artefact, not as a current tool.

## Model details

| Property | Value |
|---|---|
| Base model | [Qwen 2.5 Coder 7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| Method | QLoRA (rank 64, alpha 128, 3 epochs) |
| Target syntax | toke **v0.3** |
| Weights | AWQ 4-bit quantised |
| Context length | 32,768 tokens |
| Tokenizer | Qwen's own (151K vocab) — this model does **not** use a toke tokenizer |
| Licence | Apache-2.0 |

## Results, with the set they were measured on

**Gate 2, 2026-05-22 — curated set.** On the curated 500-hidden + 200-eval set, the model
reached **100% compile Pass@1** and **55.6% functional** (272/489).

**The honest floor — full-local re-audit.** Across all 1,748 v0.3.9 corpus programs the
same artefact compiles **37.5%** (655/1,748) and is **about 2.2% fully correct**
(38 PASS).

Never quote the 100% without the curated set it was measured on. The August 2026 corpus
work was a training-data quality freeze, not a model gate: the most recent model gate
remains Gate 2 above.

*Source: [`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
§ Correctness and § 2026-08.*

## Token efficiency

**Token efficiency, measured:** under one shared tokenizer (cl100k_base) toke costs
**1.34× [1.22, 1.48]** the tokens of equivalent Python on the 60 Gate-1 tasks (N = 60,
2026-09-19) — more, not fewer. The v0.3-era "52% fewer tokens" figure was a
*tokenizer-vs-tokenizer* measurement on identical toke text (Toke-16K v0.3 vs cl100k_base,
N = 42) and is superseded: on canonical v0.4 text the shipped 8K tokenizer needs **15.4%
more** tokens than cl100k_base (N = 2,000). See `docs/metrics-baseline.md`.

## Intended use and limitations

- **Intended:** research on constrained decoding and on code generation into a small,
  canonical target language; reproducing the Gate 2 result.
- **Not intended:** production code generation, general-purpose coding in other
  languages, or safety-critical work. Output is v0.3 syntax and must be reviewed and
  migrated (`tkc --migrate`) before it is used against a current compiler.
- The model generates toke only; prompts in other languages produce poor results.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("karwalski/toke", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("karwalski/toke")

prompt = """<|im_start|>system
Write toke programs (v0.3 syntax). m=mod; f=name(p:type):ret{body}; let x=42; <expr return.
<|im_end|>
<|im_start|>user
Write a hello world program
<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=True)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{toke-7b-gate2,
  title  = {toke-7b-gate2: a 7B model fine-tuned to generate toke (v0.3 syntax)},
  author = {Watt, Matthew},
  year   = {2026},
  url    = {https://huggingface.co/karwalski/toke}
}
```

## Contact

Open an issue at [github.com/karwalski/toke](https://github.com/karwalski/toke/issues).
