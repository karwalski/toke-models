# toke inference server — self-hosted via HuggingFace TGI
# Model: karwalski/toke (AWQ 4-bit, ~5.3 GB)
#
# The TGI image handles model download, quantised inference, and the
# /generate HTTP endpoint.  All we add are environment defaults so
# `docker compose up` works out of the box.

FROM ghcr.io/huggingface/text-generation-inference:3.0.1

# ── Model configuration ──────────────────────────────────────────────
ENV MODEL_ID=karwalski/toke
ENV QUANTIZE=awq
ENV MAX_INPUT_LENGTH=4096
ENV MAX_TOTAL_TOKENS=4608
ENV MAX_BATCH_PREFILL_TOKENS=4096
ENV PORT=80

# ── Sensible defaults (overridable at runtime) ───────────────────────
ENV MAX_CONCURRENT_REQUESTS=4
ENV HUGGING_FACE_HUB_TOKEN=""
