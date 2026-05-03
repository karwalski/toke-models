#!/usr/bin/env bash
# migrate_corpus.sh — Migrate legacy .tk corpus using toke --migrate + --check.
#
# Pipeline:
#   input .tk files -> toke --migrate -> toke --check -> validated output
#                                         | (if fails)
#                                         v
#                                         error.jsonl (for LLM repair)
#
# Usage:
#   ./migrate_corpus.sh --input corpus/legacy/ --output corpus/default/
#   ./migrate_corpus.sh --input corpus/legacy/ --output corpus/default/ --workers 8
#   ./migrate_corpus.sh --input corpus/legacy/ --output corpus/default/ --toke /path/to/toke

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

TOKE="${TOKE:-/Users/matthew.watt/tk/toke/toke}"
INPUT_DIR=""
OUTPUT_DIR=""
WORKERS=1
ERROR_JSONL=""   # auto-derived if empty

# ── Argument parsing ─────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
Usage: migrate_corpus.sh --input DIR --output DIR [OPTIONS]

Options:
  --input DIR       Input directory containing .tk files (required)
  --output DIR      Output directory for validated files (required)
  --toke PATH       Path to toke binary (default: /Users/matthew.watt/tk/toke/toke)
  --workers N       Number of parallel workers (default: 1)
  --errors PATH     Path for error JSONL output (default: <output>/error.jsonl)
  --help            Show this help
USAGE
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT_DIR="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        --toke)    TOKE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --errors)  ERROR_JSONL="$2"; shift 2 ;;
        --help)    usage 0 ;;
        *)         echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --input and --output are required." >&2
    usage 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

if [[ ! -x "$TOKE" ]]; then
    echo "Error: toke binary not found or not executable: $TOKE" >&2
    exit 1
fi

# ── Setup ────────────────────────────────────────────────────────────────────

mkdir -p "$OUTPUT_DIR"
ERROR_JSONL="${ERROR_JSONL:-${OUTPUT_DIR}/error.jsonl}"
: > "$ERROR_JSONL"   # truncate

# Counters (using temp files for parallel-safe accumulation)
TMPDIR_STATS="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_STATS"' EXIT

: > "$TMPDIR_STATS/total"
: > "$TMPDIR_STATS/migrated"
: > "$TMPDIR_STATS/passed"
: > "$TMPDIR_STATS/failed"
: > "$TMPDIR_STATS/migrate_error"


# ── Per-file processing ──────────────────────────────────────────────────────

process_file() {
    local input_file="$1"
    local input_dir="$2"
    local output_dir="$3"
    local toke="$4"
    local error_jsonl="$5"
    local stats_dir="$6"

    # Compute relative path and output location
    local rel_path="${input_file#${input_dir}/}"
    local out_file="${output_dir}/${rel_path}"
    local out_parent
    out_parent="$(dirname "$out_file")"
    mkdir -p "$out_parent"

    # Increment total (atomic via temp file append)
    echo "1" >> "$stats_dir/total"

    # Read original source
    local original
    original="$(cat "$input_file")"

    # Step 1: toke --migrate
    local migrated
    if ! migrated=$("$toke" --migrate "$input_file" 2>/dev/null); then
        echo "1" >> "$stats_dir/migrate_error"
        # Write error record: migration itself failed
        local escaped_original
        escaped_original="$(printf '%s' "$original" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
        # Atomic-enough append for small JSONL lines
        printf '{"source":"","errors":[{"message":"toke --migrate failed"}],"original":%s}\n' \
            "$escaped_original" >> "$error_jsonl"
        return
    fi

    echo "1" >> "$stats_dir/migrated"

    # Write migrated source to a temp file for --check
    local tmp_check
    tmp_check="$(mktemp "${TMPDIR_STATS}/check_XXXXXX.tk")"
    printf '%s' "$migrated" > "$tmp_check"

    # Step 2: toke --check --diag-json
    local diag_output check_rc
    diag_output=$("$toke" --check --diag-json "$tmp_check" 2>&1) && check_rc=0 || check_rc=$?

    rm -f "$tmp_check"

    if [[ $check_rc -eq 0 ]]; then
        # Passed — write to output
        echo "1" >> "$stats_dir/passed"
        printf '%s' "$migrated" > "$out_file"
    else
        # Failed — write error JSONL for LLM repair
        echo "1" >> "$stats_dir/failed"

        local escaped_source escaped_original errors_json
        escaped_source="$(printf '%s' "$migrated" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
        escaped_original="$(printf '%s' "$original" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"

        # Parse diagnostic lines into a JSON array
        if [[ -n "$diag_output" ]]; then
            errors_json="$(printf '%s' "$diag_output" | python3 -c '
import sys, json
lines = [l.strip() for l in sys.stdin if l.strip()]
arr = []
for l in lines:
    try:
        arr.append(json.loads(l))
    except json.JSONDecodeError:
        arr.append({"message": l})
print(json.dumps(arr))
')"
        else
            errors_json="[]"
        fi

        printf '{"source":%s,"errors":%s,"original":%s}\n' \
            "$escaped_source" "$errors_json" "$escaped_original" >> "$error_jsonl"
    fi
}

export -f process_file

# ── Main ─────────────────────────────────────────────────────────────────────

echo "migrate_corpus.sh"
echo "  Input:   $INPUT_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Errors:  $ERROR_JSONL"
echo "  toke:    $TOKE"
echo "  Workers: $WORKERS"
echo ""

# Collect all .tk files into a temp manifest
TK_MANIFEST="$TMPDIR_STATS/manifest.txt"
find "$INPUT_DIR" -name '*.tk' -type f | sort > "$TK_MANIFEST"

FILE_COUNT=$(wc -l < "$TK_MANIFEST" | tr -d ' ')

if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo "No .tk files found in $INPUT_DIR"
    exit 0
fi

echo "Found $FILE_COUNT .tk files"
echo ""

if [[ $WORKERS -le 1 ]]; then
    # Sequential processing
    while IFS= read -r f; do
        process_file "$f" "$INPUT_DIR" "$OUTPUT_DIR" "$TOKE" "$ERROR_JSONL" "$TMPDIR_STATS"
    done < "$TK_MANIFEST"
else
    # Parallel processing via xargs
    xargs -P "$WORKERS" -I {} bash -c \
        'process_file "$@"' _ {} "$INPUT_DIR" "$OUTPUT_DIR" "$TOKE" "$ERROR_JSONL" "$TMPDIR_STATS" \
        < "$TK_MANIFEST"
fi

# ── Report ───────────────────────────────────────────────────────────────────

count_lines() {
    wc -l < "$1" | tr -d ' '
}

TOTAL=$(count_lines "$TMPDIR_STATS/total")
MIGRATED=$(count_lines "$TMPDIR_STATS/migrated")
PASSED=$(count_lines "$TMPDIR_STATS/passed")
FAILED=$(count_lines "$TMPDIR_STATS/failed")
MIGRATE_ERR=$(count_lines "$TMPDIR_STATS/migrate_error")

echo ""
echo "── Results ──────────────────────────────────"
echo "  Total files:       $TOTAL"
echo "  Migrated OK:       $MIGRATED"
echo "  --check passed:    $PASSED"
echo "  --check failed:    $FAILED"
echo "  --migrate errors:  $MIGRATE_ERR"
if [[ $MIGRATED -gt 0 ]]; then
    PASS_RATE=$(python3 -c "print(f'{$PASSED/$MIGRATED*100:.1f}%')")
    echo "  Pass rate:         $PASS_RATE"
fi
echo ""

if [[ $FAILED -gt 0 || $MIGRATE_ERR -gt 0 ]]; then
    ERROR_COUNT=$((FAILED + MIGRATE_ERR))
    echo "$ERROR_COUNT error(s) written to: $ERROR_JSONL"
fi

echo "Validated output in: $OUTPUT_DIR"
