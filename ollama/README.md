# Publishing toke to Ollama

## Registry description (story 132.4 — not yet published)

`ollama.com/library/karwalski/toke` **does not exist yet** (checked 2026-09-19: 404).
Nothing is live to correct; the text below is what the description field must say when
the owner first pushes. It is the canonical one-liner plus the two qualifiers this model
cannot be published without.

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification. This is the Gate 2 research artefact: Qwen
> 2.5 Coder 7B + QLoRA, fine-tuned on toke **v0.3** syntax, so its output does not match
> the current v0.4 specification. Gate 2 (2026-05-22) measured 100% compile Pass@1 and
> 55.6% functional on a curated 500-hidden + 200-eval set; across all 1,748 v0.3.9 corpus
> programs the same artefact compiles 37.5%. Details: tokelang.dev ·
> github.com/karwalski/toke

**Two things travel with this model and must not be dropped:** the 100% is a *curated
set* number, and the model writes v0.3, not v0.4.

**The `Modelfile` SYSTEM prompt is v0.3 too** — it lists the superseded v0.3 keyword set (13 of them, no `sc`), `$`-prefixed
types and v0.3 array syntax. That is correct *for this model*, which was trained on that
syntax, and it must not be "fixed" to v0.4 wording: changing the prompt would describe a
language the weights have never seen. It is why the description has to name v0.3
explicitly. The current language has 14 keywords — see
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).

## Conversion and publishing

Converts the toke 7B Gate 2 model from HuggingFace safetensors format to GGUF
and publishes it to the Ollama registry as `karwalski/toke`.

## Prerequisites

- Python 3.10+
- `pip install huggingface-hub`
- ~16 GB disk space for intermediate files
- Ollama installed (`brew install ollama` or https://ollama.com)

## Step-by-step

### 1. Clone llama.cpp and build quantize tool

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build --config Release -j
cd ..
```

### 2. Download the model from HuggingFace

```bash
huggingface-cli download karwalski/toke --local-dir ./hf-model
```

### 3. Convert to GGUF (float16)

```bash
python llama.cpp/convert_hf_to_gguf.py ./hf-model --outfile toke-7b-f16.gguf --outtype f16
```

### 4. Quantize to Q4_K_M

Q4_K_M gives a good balance of quality and size for 7B models (~4.5 GB).

```bash
./llama.cpp/build/bin/llama-quantize toke-7b-f16.gguf toke-7b-gate2-q4km.gguf Q4_K_M
```

### 5. Create the Ollama model

The `Modelfile` in this directory has the system prompt baked in and uses
ChatML template format matching Qwen 2.5.

```bash
ollama create karwalski/toke -f Modelfile
```

### 6. Test locally

```bash
ollama run karwalski/toke "return the absolute value of an integer"
```

### 7. Push to Ollama registry

Requires an Ollama account with the `karwalski` namespace.

```bash
ollama push karwalski/toke
```

## Automated conversion

The `convert.sh` script automates steps 1-5:

```bash
chmod +x convert.sh
./convert.sh           # convert and create ollama model
./convert.sh --push    # also push to ollama registry
```

## Usage

Once published, anyone can run:

```bash
ollama run karwalski/toke "fibonacci sequence up to n"
```

## Model details

| Property     | Value                            |
|------------- |--------------------------------- |
| Base model   | Qwen 2.5 Coder 7B               |
| Fine-tune    | QLoRA on toke corpus             |
| Quantization | Q4_K_M (GGUF)                   |
| Size         | ~4.5 GB                          |
| HuggingFace  | karwalski/toke                   |
| Template     | ChatML (`<\|im_start\|>` format) |
