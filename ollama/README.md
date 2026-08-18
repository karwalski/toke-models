# Publishing toke to Ollama

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
