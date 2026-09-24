#!/usr/bin/env bash
set -euo pipefail
#
# Runs on the training instance, inside tmux, started by cloud/aws_train.sh:
# train, then hand the results to cloud/post_train.sh (the one post-train
# script; story 132.82).
#
# Cost guard (132.82). A GPU instance must never be left running because a
# step failed quietly:
#   - a dead-man's switch schedules a shutdown MAX_RUNTIME_MIN minutes after
#     start (default 720), so even a hung run ends;
#   - an EXIT trap: if anything fails — training, packaging, upload, push —
#     the shutdown is brought forward to FAILURE_GRACE_MIN minutes (default
#     30), time to ssh in and cancel it with `sudo shutdown -c`.
# On success post_train.sh schedules its own shutdown one minute out.
# On a one-time spot instance, shutdown terminates the instance and its
# volume: anything not uploaded is lost once the grace period ends.
#
# Opt-outs, both explicit:
#   KEEP_ALIVE_ON_FAILURE=1   the trap only logs; the instance stays up
#   MAX_RUNTIME_MIN=0         no dead-man's switch
#
# Settings come from cloud/post_train.env, written by aws_train.sh.

# This script lives in $WORK_DIR/cloud/.
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$WORK_DIR/cloud/post_train.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-720}"
FAILURE_GRACE_MIN="${FAILURE_GRACE_MIN:-30}"
KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-finetune/configs/7b.yaml}"
LOG="$WORK_DIR/train.log"

log() { echo "[$(date '+%H:%M:%S')] RUN: $*" | tee -a "$LOG"; }

on_exit() {
    local code=$?
    rm -f "$HOME/.git-credentials"
    if [[ $code -eq 0 ]]; then
        return
    fi
    log "=== FAILED with exit code $code ==="
    if [[ "$KEEP_ALIVE_ON_FAILURE" == 1 ]]; then
        log "KEEP_ALIVE_ON_FAILURE=1: instance stays UP and keeps billing."
        if [[ "$MAX_RUNTIME_MIN" != 0 ]]; then
            log "The dead-man's switch still shuts it down $MAX_RUNTIME_MIN min after start."
        fi
        return
    fi
    log "Instance shuts down in $FAILURE_GRACE_MIN min. To keep it: sudo shutdown -c"
    sudo shutdown -h "+$FAILURE_GRACE_MIN" "toke training failed (exit $code)" || true
}
trap on_exit EXIT

cd "$WORK_DIR"
if [[ "$MAX_RUNTIME_MIN" != 0 ]]; then
    sudo shutdown -h "+$MAX_RUNTIME_MIN" "toke training: dead-man's switch"
    log "Dead-man's switch: shutdown in $MAX_RUNTIME_MIN min unless the run finishes first."
fi

# shellcheck disable=SC1091
source "$HOME/toke-venv/bin/activate"

log "=== Training started at $(date -u) ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"

python3 finetune/train_qlora.py --config "$TRAIN_CONFIG" 2>&1 | tee -a "$LOG"
log "=== Training finished at $(date -u) ==="

log "=== Starting post-training upload ==="
bash cloud/post_train.sh 2>&1 | tee -a "$LOG"
log "=== Post-training complete ==="
