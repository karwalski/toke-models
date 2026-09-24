#!/usr/bin/env bash
set -euo pipefail
#
# Package a finished training run.
#
# Adapter weights go to the model store, never to git (workspace AGENTS.md
# §4.4, story 132.77). What is committed is a small pointer file naming the
# stored object, its SHA-256 and its size, plus the metrics JSON and a README.
#
# Required when an adapter exists:
#   MODEL_STORE_URI   destination prefix, e.g. s3://<bucket>/<prefix>
#                     (uploaded with `aws s3 cp`; the instance needs write
#                     access to it)
# Optional:
#   WORK_DIR, REPO_DIR       override the default instance paths
#   POST_TRAIN_PUSH=0        commit locally but do not push
#   POST_TRAIN_SHUTDOWN=0    do not shut the instance down afterwards
#
# This is the only post-training script: cloud/aws_train.sh uploads this
# file, and cloud/run_training.sh runs it after training (story 132.82).
# A non-zero exit here trips run_training.sh's cost guard, which shuts the
# instance down after a grace period.

WORK_DIR="${WORK_DIR:-/home/ubuntu/toke-model}"
REPO_DIR="${REPO_DIR:-/home/ubuntu/toke-model-repo}"
GITHUB_REPO="${GITHUB_REPO:-karwalski/toke-models}"
MODEL_STORE_URI="${MODEL_STORE_URI:-}"
BRANCH="train-results-$(date -u +%Y%m%d-%H%M%S)"
MAX_COMMIT_BYTES=$((5 * 1024 * 1024))

log() { echo "[$(date '+%H:%M:%S')] POST-TRAIN: $*" | tee -a "$WORK_DIR/train.log"; }
die() { log "ERROR: $*"; exit 1; }

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

log "Packaging results for branch $BRANCH..."

RESULTS_REL="results/$BRANCH"
RESULTS_DIR="$REPO_DIR/$RESULTS_REL"
STAGE_DIR="$WORK_DIR/output/store-upload/$BRANCH"   # outside the repo
mkdir -p "$RESULTS_DIR" "$STAGE_DIR"

# Small metrics files — these are committed.
cp "$WORK_DIR/output/7b-qlora/training_metrics.json" "$RESULTS_DIR/" 2>/dev/null || true
cp "$WORK_DIR/output/7b-qlora/eval_metrics.json" "$RESULTS_DIR/" 2>/dev/null || true
# The full log can be large: it goes to the store with the weights.
cp "$WORK_DIR/train.log" "$STAGE_DIR/" 2>/dev/null || true

ADAPTER_DIR="$WORK_DIR/output/7b-qlora/adapter"
POINTER="$RESULTS_DIR/adapter-weights.pointer.json"
if [[ -d "$ADAPTER_DIR" ]]; then
    [[ -n "$MODEL_STORE_URI" ]] || die "adapter found but MODEL_STORE_URI is unset; weights are not committed to git"
    TARBALL="$STAGE_DIR/adapter-weights.tar.gz"
    log "Compressing adapter weights..."
    (cd "$WORK_DIR/output/7b-qlora" && tar czf "$TARBALL" adapter/)
    DEST="${MODEL_STORE_URI%/}/$BRANCH"
    log "Uploading adapter to $DEST/ ..."
    aws s3 cp "$TARBALL" "$DEST/adapter-weights.tar.gz"
    if [[ -f "$STAGE_DIR/train.log" ]]; then
        aws s3 cp "$STAGE_DIR/train.log" "$DEST/train.log"
    fi
    cat > "$POINTER" <<POINTER_EOF
{
  "uri": "$DEST/adapter-weights.tar.gz",
  "sha256": "$(sha256_of "$TARBALL")",
  "bytes": $(wc -c < "$TARBALL" | tr -d ' '),
  "log_uri": "$DEST/train.log",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
POINTER_EOF
    log "Pointer written: $RESULTS_REL/adapter-weights.pointer.json"
else
    log "WARNING: no adapter directory at $ADAPTER_DIR; committing metrics only"
fi

cat > "$RESULTS_DIR/README.md" <<README
# Training Run: $BRANCH

- **Model:** Qwen 2.5 Coder 7B Instruct
- **Method:** QLoRA (4-bit NF4, rank 64, alpha 128)
- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
- **Date:** $(date -u +%Y-%m-%d)

## Files

- \`adapter-weights.pointer.json\` — where the LoRA adapter lives in the model
  store (URI, SHA-256, size). Fetch it, verify the hash, then merge with the
  base model using merge_adapters.py. The weights are never in git.
- \`training_metrics.json\` — loss curve and training stats
- \`eval_metrics.json\` — evaluation metrics
- \`train.log\` — stored beside the adapter (\`log_uri\` in the pointer)
README

cd "$REPO_DIR"
git checkout -b "$BRANCH"
git config user.email "training@toke.dev"
git config user.name "toke-training-bot"

# results/ is gitignored, so name each small file explicitly and force-add
# only those. Never `git add results/`.
TO_ADD=()
for f in README.md training_metrics.json eval_metrics.json adapter-weights.pointer.json; do
    [[ -f "$RESULTS_DIR/$f" ]] && TO_ADD+=("$RESULTS_REL/$f")
done
git add -f -- "${TO_ADD[@]}"

# Gate: nothing weight-like and nothing over 5 MB may be committed.
while IFS= read -r p; do
    [[ -z "$p" ]] && continue
    case "$p" in
        *.tar.gz|*.tgz|*.tar|*.safetensors|*.bin|*.gguf|*.pt|*.pth|*.ckpt|*.npz)
            die "refusing to commit weight-like file $p" ;;
    esac
    sz=$(git cat-file -s ":$p")
    (( sz <= MAX_COMMIT_BYTES )) || die "refusing to commit $p ($sz bytes > 5 MB)"
done < <(git diff --cached --name-only)

git commit -m "training: QLoRA 7B run $BRANCH

Adapter weights stored at the URI in adapter-weights.pointer.json."

if [[ "${POST_TRAIN_PUSH:-1}" != 0 ]]; then
    log "Pushing results to GitHub branch $BRANCH..."
    git push origin "$BRANCH" 2>&1 | tail -5
    log "Results pushed to: https://github.com/$GITHUB_REPO/tree/$BRANCH"
fi

if [[ "${POST_TRAIN_SHUTDOWN:-1}" != 0 ]]; then
    log "Shutting down instance in 60 seconds..."
    sudo shutdown -h +1 "Training complete. Results pushed to GitHub."
fi
