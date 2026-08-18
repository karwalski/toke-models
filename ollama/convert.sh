#!/usr/bin/env bash
# convert.sh — Convert toke HuggingFace model to GGUF and create Ollama model
# Usage: ./convert.sh [--push]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
HF_MODEL_DIR="${WORK_DIR}/hf-model"
LLAMA_CPP_DIR="${WORK_DIR}/llama.cpp"
F16_GGUF="${WORK_DIR}/toke-7b-f16.gguf"
Q4KM_GGUF="${WORK_DIR}/toke-7b-gate2-q4km.gguf"
MODELFILE="${WORK_DIR}/Modelfile"
OLLAMA_MODEL="karwalski/toke"
PUSH=false

if [[ "${1:-}" == "--push" ]]; then
    PUSH=true
fi

echo "=== toke GGUF conversion pipeline ==="
echo "Working directory: ${WORK_DIR}"
echo ""

# --- Step 1: Clone llama.cpp if not present ---
if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    echo "[1/5] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
else
    echo "[1/5] llama.cpp already present, pulling latest..."
    git -C "${LLAMA_CPP_DIR}" pull --ff-only || true
fi

# --- Step 2: Build llama.cpp quantize tool ---
echo "[2/5] Building llama.cpp..."
cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
if [[ ! -x "${QUANTIZE_BIN}" ]]; then
    echo "ERROR: llama-quantize not found at ${QUANTIZE_BIN}"
    exit 1
fi

# --- Step 3: Download model from HuggingFace ---
if [[ ! -d "${HF_MODEL_DIR}" ]] || [[ ! -f "${HF_MODEL_DIR}/config.json" ]]; then
    echo "[3/5] Downloading model from HuggingFace (karwalski/toke)..."
    huggingface-cli download karwalski/toke --local-dir "${HF_MODEL_DIR}"
else
    echo "[3/5] HuggingFace model already downloaded, skipping."
fi

# --- Step 4: Convert to GGUF float16 ---
if [[ ! -f "${F16_GGUF}" ]]; then
    echo "[4/5] Converting to GGUF (float16)..."
    python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "${HF_MODEL_DIR}" \
        --outfile "${F16_GGUF}" \
        --outtype f16
else
    echo "[4/5] Float16 GGUF already exists, skipping conversion."
fi

# --- Step 5: Quantize to Q4_K_M ---
if [[ ! -f "${Q4KM_GGUF}" ]]; then
    echo "[5/5] Quantizing to Q4_K_M..."
    "${QUANTIZE_BIN}" "${F16_GGUF}" "${Q4KM_GGUF}" Q4_K_M
else
    echo "[5/5] Q4_K_M GGUF already exists, skipping quantization."
fi

echo ""
echo "=== Creating Ollama model ==="
ollama create "${OLLAMA_MODEL}" -f "${MODELFILE}"

echo ""
echo "=== Testing model ==="
echo "Running: ollama run ${OLLAMA_MODEL} 'hello world program'"
ollama run "${OLLAMA_MODEL}" "hello world program"

if [[ "${PUSH}" == true ]]; then
    echo ""
    echo "=== Pushing to Ollama registry ==="
    ollama push "${OLLAMA_MODEL}"
    echo "Done. Model available at: https://ollama.com/karwalski/toke"
else
    echo ""
    echo "Skipping push. Run with --push to publish to Ollama registry."
fi

echo ""
echo "=== Summary ==="
echo "Float16 GGUF: ${F16_GGUF}"
echo "Q4_K_M GGUF:  ${Q4KM_GGUF}"
echo "Ollama model: ${OLLAMA_MODEL}"
ls -lh "${Q4KM_GGUF}" 2>/dev/null || true
