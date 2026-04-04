#!/usr/bin/env bash
# upload.sh — Upload toke-coder-7b model to Hugging Face Hub
# Story 6.1.2: Upload model weights and tokenizer
#
# Usage:
#   ./upload.sh              # validate and upload
#   ./upload.sh --dry-run    # validate only, no upload

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
REPO_ID="karwalski/toke-coder-7b"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths to artifacts
ADAPTER_DIR="$MODELS_DIR/results/train-results-20260402-055104/adapter-mlx"
MERGED_DIR="$MODELS_DIR/output/7b-merged"
HF_DIR="$SCRIPT_DIR"

# ── Parse flags ───────────────────────────────────────────────
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE — validating only, no upload ==="
    echo
fi

# ── Helpers ───────────────────────────────────────────────────
fail=0

check_file() {
    local path="$1"
    local label="$2"
    if [[ -f "$path" ]]; then
        printf "  ✓ %-40s %s\n" "$label" "$(du -h "$path" | cut -f1)"
    else
        printf "  ✗ %-40s MISSING\n" "$label"
        fail=1
    fi
}

check_dir() {
    local path="$1"
    local label="$2"
    if [[ -d "$path" ]]; then
        local count
        count=$(find "$path" -maxdepth 1 -type f | wc -l | tr -d ' ')
        printf "  ✓ %-40s %s files\n" "$label" "$count"
    else
        printf "  ✗ %-40s MISSING\n" "$label"
        fail=1
    fi
}

# ── Step 1: Validate prerequisites ───────────────────────────
echo "Checking prerequisites..."

if ! command -v huggingface-cli &>/dev/null; then
    echo "  ✗ huggingface-cli not found. Install with: pip install huggingface-hub"
    fail=1
else
    echo "  ✓ huggingface-cli found"
fi

if ! command -v git-lfs &>/dev/null && ! git lfs version &>/dev/null 2>&1; then
    echo "  ✗ git-lfs not found. Install with: brew install git-lfs"
    fail=1
else
    echo "  ✓ git-lfs found"
fi

echo

# ── Step 2: Validate HuggingFace directory files ─────────────
echo "Checking HuggingFace directory files..."
check_file "$HF_DIR/README.md"              "README.md (model card)"
check_file "$HF_DIR/config.json"            "config.json"
check_file "$HF_DIR/tokenizer_config.json"  "tokenizer_config.json"
check_file "$HF_DIR/.gitattributes"         ".gitattributes (LFS tracking)"
echo

# ── Step 3: Validate merged model weights ────────────────────
echo "Checking merged model weights..."
check_dir  "$MERGED_DIR"                              "Merged model directory"
check_file "$MERGED_DIR/model.safetensors"            "model.safetensors"
check_file "$MERGED_DIR/config.json"                  "config.json (model)"
check_file "$MERGED_DIR/tokenizer.json"               "tokenizer.json"
check_file "$MERGED_DIR/tokenizer_config.json"        "tokenizer_config.json"
check_file "$MERGED_DIR/generation_config.json"       "generation_config.json"
echo

# ── Step 4: Validate adapter weights (optional) ──────────────
echo "Checking adapter weights (optional, for adapter-only upload)..."
check_dir  "$ADAPTER_DIR"                             "Adapter directory"
echo

# ── Validation result ────────────────────────────────────────
if [[ $fail -ne 0 ]]; then
    echo "VALIDATION FAILED — fix missing files before uploading."
    exit 1
fi

echo "All required files present."
echo

if $DRY_RUN; then
    echo "Dry run complete. To upload, run without --dry-run."
    exit 0
fi

# ── Step 5: Create repo if it doesn't exist ──────────────────
echo "Creating HuggingFace repo (if it doesn't exist)..."
huggingface-cli repo create "$(basename "$REPO_ID")" \
    --organization "$(dirname "$REPO_ID")" \
    --type model \
    --exist-ok \
    || echo "  (repo may already exist, continuing)"
echo

# ── Step 6: Upload files ─────────────────────────────────────
echo "Uploading to $REPO_ID..."

# Upload merged model directory (includes weights, tokenizer, configs)
huggingface-cli upload "$REPO_ID" "$MERGED_DIR" . \
    --repo-type model \
    --commit-message "Upload merged model weights and tokenizer"

# Upload HuggingFace-specific files (model card, custom config, etc.)
huggingface-cli upload "$REPO_ID" "$HF_DIR/README.md" README.md \
    --repo-type model \
    --commit-message "Upload model card"

huggingface-cli upload "$REPO_ID" "$HF_DIR/config.json" config.json \
    --repo-type model \
    --commit-message "Upload toke-specific config"

huggingface-cli upload "$REPO_ID" "$HF_DIR/tokenizer_config.json" tokenizer_config.json \
    --repo-type model \
    --commit-message "Upload tokenizer config"

echo
echo "Upload complete!"
echo "View at: https://huggingface.co/$REPO_ID"
