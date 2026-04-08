#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/home/ubuntu/toke-model"
REPO_DIR="/home/ubuntu/toke-model-repo"
BRANCH="train-results-$(date -u +%Y%m%d-%H%M%S)"

log() { echo "[$(date '+%H:%M:%S')] POST-TRAIN: $*" | tee -a "$WORK_DIR/train.log"; }

log "Packaging results for branch $BRANCH..."

cd "$REPO_DIR"
git checkout -b "$BRANCH"

# Create results directory
RESULTS_DIR="$REPO_DIR/results/$BRANCH"
mkdir -p "$RESULTS_DIR"

# Copy metrics and log (small files — safe for git)
cp "$WORK_DIR/output/7b-qlora/training_metrics.json" "$RESULTS_DIR/" 2>/dev/null || true
cp "$WORK_DIR/output/7b-qlora/eval_metrics.json" "$RESULTS_DIR/" 2>/dev/null || true
cp "$WORK_DIR/train.log" "$RESULTS_DIR/" 2>/dev/null || true

# Tar adapter weights (will be pushed via LFS or as a binary)
ADAPTER_DIR="$WORK_DIR/output/7b-qlora/adapter"
if [[ -d "$ADAPTER_DIR" ]]; then
    log "Compressing adapter weights..."
    cd "$WORK_DIR/output/7b-qlora"
    tar czf "$RESULTS_DIR/adapter-weights.tar.gz" adapter/
    ADAPTER_SIZE=$(du -sh "$RESULTS_DIR/adapter-weights.tar.gz" | cut -f1)
    log "Adapter archive: $ADAPTER_SIZE"
fi

# Build a summary README
cat > "$RESULTS_DIR/README.md" <<README
# Training Run: $BRANCH

- **Model:** Qwen 2.5 Coder 7B Instruct
- **Method:** QLoRA (4-bit NF4, rank 64, alpha 128)
- **Corpus:** 46,754 toke programs → 73,026 training examples
- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
- **Date:** $(date -u +%Y-%m-%d)

## Results

$(if [[ -f "$RESULTS_DIR/training_metrics.json" ]]; then
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/training_metrics.json'))
print(f'- **Train loss:** {d.get(\"train_loss\", \"N/A\"):.4f}')
print(f'- **Train runtime:** {d.get(\"train_runtime\", 0)/3600:.1f} hours')
print(f'- **Train samples/sec:** {d.get(\"train_samples_per_second\", \"N/A\")}')
" 2>/dev/null || echo '- Metrics: see training_metrics.json'
fi)

$(if [[ -f "$RESULTS_DIR/eval_metrics.json" ]]; then
    python3 -c "
import json
d = json.load(open('$RESULTS_DIR/eval_metrics.json'))
print(f'- **Eval loss:** {d.get(\"eval_loss\", \"N/A\"):.4f}')
" 2>/dev/null || echo '- Eval: see eval_metrics.json'
fi)

## Files

- \`adapter-weights.tar.gz\` — LoRA adapter (merge with base using merge_adapters.py)
- \`training_metrics.json\` — Loss curve and training stats
- \`eval_metrics.json\` — Evaluation metrics
- \`train.log\` — Full training log
README

# Configure git
cd "$REPO_DIR"
git config user.email "training@toke.dev"
git config user.name "toke-training-bot"

# Stage and commit
git add results/
git commit -m "training: QLoRA 7B run $BRANCH

Corpus: 46,754 entries, 73,026 training examples
Config: NF4 4-bit, LoRA rank 64, 3 epochs, lr 2e-4"

# Push
log "Pushing results to GitHub branch $BRANCH..."
git push origin "$BRANCH" 2>&1 | tail -5

log "Results pushed to: https://github.com/karwalski/toke-model/tree/$BRANCH"
log "Shutting down instance in 60 seconds..."
sudo shutdown -h +1 "Training complete. Results pushed to GitHub."
