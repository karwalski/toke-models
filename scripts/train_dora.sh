#!/usr/bin/env bash
# Train DoRA adapter using the comparison config and evaluate.
# Story 10.7.2: Switch training to DoRA
#
# Usage:
#   ./scripts/train_dora.sh
#   ./scripts/train_dora.sh --resume   # resume from last checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_DIR/finetune/configs/7b_dora_comparison.yaml"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/dora_training.log"
DORA_ADAPTER_DIR="$REPO_DIR/adapters/dora"
QLORA_ADAPTER_DIR="$REPO_DIR/output/7b-mlx/adapter"
BENCHMARK_DIR="$REPO_DIR/eval/benchmark"
COMPARISON_DIR="$REPO_DIR/results/comparison"

mkdir -p "$LOG_DIR" "$DORA_ADAPTER_DIR"

echo "=== DoRA Training Pipeline ==="
echo "Config:  $CONFIG"
echo "Log:     $LOG_FILE"
echo "Adapter: $DORA_ADAPTER_DIR"
echo ""

# Parse arguments
RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume $DORA_ADAPTER_DIR"
    echo "Resuming from $DORA_ADAPTER_DIR"
fi

# Step 1: Train DoRA adapter
echo "[1/3] Training DoRA adapter ..."
cd "$REPO_DIR"
python finetune/train_mlx.py --config "$CONFIG" $RESUME_FLAG 2>&1 | tee "$LOG_FILE"

echo ""
echo "[2/3] Evaluating DoRA adapter on benchmark tasks ..."

# Step 2: Run evaluation if benchmark dir and eval script exist
if [[ -f "$SCRIPT_DIR/eval_adapter.py" ]]; then
    python "$SCRIPT_DIR/eval_adapter.py" \
        --adapter-dir "$DORA_ADAPTER_DIR" \
        --benchmark-dir "$BENCHMARK_DIR" \
        --output "$DORA_ADAPTER_DIR/predictions.jsonl" \
        --model-base "Qwen/Qwen2.5-Coder-7B-Instruct" \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "WARNING: eval_adapter.py not found, skipping evaluation" | tee -a "$LOG_FILE"
fi

# Step 3: Compare against QLoRA baseline if available
echo ""
echo "[3/3] Comparing against QLoRA baseline ..."

if [[ -d "$QLORA_ADAPTER_DIR" ]] && [[ -f "$QLORA_ADAPTER_DIR/predictions.jsonl" ]]; then
    mkdir -p "$COMPARISON_DIR"
    python "$SCRIPT_DIR/compare_dora_qlora.py" \
        --qlora-dir "$QLORA_ADAPTER_DIR" \
        --dora-dir "$DORA_ADAPTER_DIR" \
        --benchmark-dir "$BENCHMARK_DIR" \
        --output-dir "$COMPARISON_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    echo ""
    echo "Comparison report: $COMPARISON_DIR/comparison_report.md"
else
    echo "QLoRA baseline predictions not found at $QLORA_ADAPTER_DIR/predictions.jsonl"
    echo "Run evaluation on QLoRA adapter first, then re-run comparison."
fi | tee -a "$LOG_FILE"

echo ""
echo "=== DoRA pipeline complete ==="
echo "Training log: $LOG_FILE"
echo "Adapter: $DORA_ADAPTER_DIR"
