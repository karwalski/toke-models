# toke self-hosted inference

Run the toke code-generation model locally using HuggingFace
Text Generation Inference (TGI).

The model (`karwalski/toke`) is an AWQ 4-bit quantised 7B parameter model
trained to generate toke source code.

## Quick start (GPU)

```bash
docker compose up
```

First run downloads ~5.3 GB of model weights from HuggingFace.
Subsequent starts use the cached volume and are fast.

### Requirements

- Docker with Compose v2
- NVIDIA GPU with 8 GB+ VRAM (e.g. RTX 3070 or better)
- `nvidia-container-toolkit` installed
  ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

## CPU-only mode

```bash
docker compose -f docker-compose.cpu.yml up
```

No GPU required, but inference is slow (~30-60 seconds per response).
Useful for testing the API integration without GPU hardware.

## API usage

TGI exposes its native HTTP API on port 8080 (mapped from container port 80).

### Generate toke code

```bash
curl -s http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": "<|im_start|>system\nWrite toke programs. Lowercase only. Semicolons separate statements. No commas. No square brackets.\nKEYWORDS: m f t i if el lp br let mut as rt mt\nBare i= is import, f= is function, t= is type, m= is module.\nAll types use $ prefix: $i64, $str, $bool, $mytype, $point. Always $.\nLoop: lp(let idx=0;idx<n;idx=idx+1){body}. Never lp(i=...) \u2014 i= is import.\nFunctions: f=name(p:$i64):$i64{body}. Return: <expr; or rt expr;\nTypes: t=point{x:$i64;y:$i64}. Arrays: @(1;2;3); arr.get(idx);\nImports: i=io:std.io; Conditionals: if(cond){body}el{body}\nFib: m=fib;f=fib(n:$i64):$i64{if(n<2){<n};let a=mut.0;let b=mut.1;lp(let idx=2;idx<n;idx=idx+1){let tmp=a+b;a=b;b=tmp};<b}\nOutput ONE complete program. Do not repeat functions. Start with m=.<|im_end|>\n<|im_start|>user\nwrite a function that adds two numbers<|im_end|>\n<|im_start|>assistant\n",
    "parameters": {
      "max_new_tokens": 256,
      "temperature": 0.2,
      "do_sample": true,
      "return_full_text": false
    }
  }'
```

### Health check

```bash
curl http://localhost:8080/health
```

Returns HTTP 200 when the model is loaded and ready.

### Response format

```json
[
  {
    "generated_text": "m=adder;f=add(a:$i64;b:$i64):$i64{<a+b}"
  }
]
```

## Using with toke MCP server

Point the MCP server at your local instance:

```bash
export TOKE_API_URL=http://localhost:8080
```

The MCP server will use TGI's `/generate` endpoint directly instead of the
cloud API.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `karwalski/toke` | HuggingFace model repository |
| `QUANTIZE` | `awq` | Quantisation method (GPU only) |
| `MAX_INPUT_LENGTH` | `4096` | Maximum input token length |
| `MAX_TOTAL_TOKENS` | `4608` | Max input + output tokens |
| `MAX_BATCH_PREFILL_TOKENS` | `4096` | Prefill batch budget |
| `MAX_CONCURRENT_REQUESTS` | `4` | Concurrent request limit |
| `HUGGING_FACE_HUB_TOKEN` | (empty) | HF token if model is gated |

## Notes

- The model uses ChatML prompt format (`<|im_start|>` / `<|im_end|>` markers).
- TGI's native API is used directly; no additional wrapper is needed.
- The `QUANTIZE=awq` flag is only relevant on GPU; CPU mode ignores it and
  runs unquantised (which is why it needs more memory and is slower).
- Model weights are stored in the `model-cache` Docker volume at `/data`.
  To force a re-download, remove the volume: `docker volume rm docker_model-cache`.
