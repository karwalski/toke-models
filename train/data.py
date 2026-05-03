"""Streaming data loader for toke 1B model training.

Reads JSONL corpus files or directories of .tk files, tokenizes with
BPE (SentencePiece) or character-level fallback, and yields padded
batches of (input_ids, target_ids) for causal language modelling.

Memory-efficient: streams from disk with buffer-based shuffling.
Returns mlx.core.array when MLX is available, else numpy arrays.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# MLX / numpy backend selection
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False

# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class _CharTokenizer:
    """Character-level fallback tokenizer (each char = one token ID)."""

    def __init__(self) -> None:
        # Reserve 0=PAD, 1=BOS, 2=EOS; printable ASCII starts at offset 3.
        self._offset = 3

    def encode(self, text: str) -> list[int]:
        return [ord(c) + self._offset for c in text]

    def vocab_size(self) -> int:
        # 3 special tokens + 256 possible byte values
        return 259


class _SPTokenizer:
    """SentencePiece BPE tokenizer wrapper."""

    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self._sp.Encode(text)  # type: ignore[no-any-return]

    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()  # type: ignore[no-any-return]


def _load_tokenizer(
    tokenizer_path: str | Path | None,
) -> _SPTokenizer | _CharTokenizer:
    """Load a SentencePiece tokenizer or fall back to character-level."""
    if tokenizer_path is not None:
        p = Path(tokenizer_path)
        # Accept either a direct .model file or a directory containing toke.model
        if p.is_file():
            return _SPTokenizer(p)
        model_file = p / "toke.model"
        if model_file.is_file():
            return _SPTokenizer(model_file)
        # Explicit path given but no model found — fall back to char tokenizer
        # without searching the default location.
        return _CharTokenizer()

    # No path given: search default location relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    default = repo_root / "tokenizer" / "models" / "toke.model"
    if default.is_file():
        return _SPTokenizer(default)

    # Fallback
    return _CharTokenizer()


# ---------------------------------------------------------------------------
# Corpus iterators
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one record per line from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_tk_files(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a directory of .tk files."""
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if name.endswith(".tk"):
                fp = Path(root) / name
                text = fp.read_text(encoding="utf-8")
                yield {"text": text, "category": fp.parent.name, "path": str(fp)}


def _iter_corpus(corpus_path: Path) -> Iterator[dict[str, Any]]:
    """Iterate over a JSONL file or directory of .tk files."""
    if corpus_path.is_file() and corpus_path.suffix in (".jsonl", ".jsonlines"):
        yield from _iter_jsonl(corpus_path)
    elif corpus_path.is_dir():
        yield from _iter_tk_files(corpus_path)
    else:
        raise ValueError(
            f"corpus_path must be a .jsonl file or directory, got: {corpus_path}"
        )


def _extract_text(record: dict[str, Any]) -> str:
    """Extract the text payload from a corpus record."""
    # Support both "text" and "code" fields
    if "text" in record:
        return record["text"]
    if "code" in record:
        return record["code"]
    raise KeyError(f"Record has neither 'text' nor 'code' field: {list(record.keys())}")


def _extract_category(record: dict[str, Any]) -> str:
    """Extract a category string for curriculum weighting."""
    if "category" in record:
        return record["category"]
    if "type" in record:
        return record["type"]
    if "path" in record:
        return Path(record["path"]).parent.name
    return "default"


# ---------------------------------------------------------------------------
# Batch construction helpers
# ---------------------------------------------------------------------------

def _pad_batch(
    sequences: list[list[int]], max_len: int, pad_id: int = PAD_ID
) -> np.ndarray:
    """Pad a list of token-id lists to max_len, return (B, max_len) array."""
    batch = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = seq
    return batch


def _to_array(arr: np.ndarray) -> Any:
    """Convert numpy array to mlx.core.array if available."""
    if _HAS_MLX:
        return mx.array(arr)
    return arr


# ---------------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------------

class TokeDataLoader:
    """Streaming data loader for toke model training.

    Parameters
    ----------
    corpus_path : str or Path
        Path to a .jsonl corpus file or a directory of .tk source files.
    tokenizer_path : str, Path, or None
        Path to SentencePiece .model file or directory containing toke.model.
        Falls back to character-level tokenizer when None and no default found.
    batch_size : int
        Number of sequences per batch.
    max_seq_len : int
        Maximum sequence length (longer sequences are truncated).
    shuffle_buffer : int
        Number of programs to buffer for shuffling (0 to disable).
    curriculum_weights : dict or None
        Mapping of category -> sampling weight.  None = uniform.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        tokenizer_path: str | Path | None = None,
        batch_size: int = 32,
        max_seq_len: int = 2048,
        shuffle_buffer: int = 10_000,
        curriculum_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.curriculum_weights = curriculum_weights
        self.seed = seed

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {self.corpus_path}")

        self.tokenizer = _load_tokenizer(tokenizer_path)

        # Cache approximate size for __len__
        self._approx_count: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text and wrap with BOS/EOS."""
        ids = self.tokenizer.encode(text)
        # Truncate to leave room for BOS + EOS
        max_body = self.max_seq_len - 2
        if len(ids) > max_body:
            ids = ids[:max_body]
        return [BOS_ID] + ids + [EOS_ID]

    def _count_programs(self) -> int:
        """Count programs in the corpus (cached)."""
        if self._approx_count is not None:
            return self._approx_count

        count = 0
        if self.corpus_path.is_file():
            with open(self.corpus_path, encoding="utf-8") as f:
                for _ in f:
                    count += 1
        else:
            for _root, _dirs, files in os.walk(self.corpus_path):
                count += sum(1 for n in files if n.endswith(".tk"))

        self._approx_count = count
        return count

    def _should_keep(self, category: str, rng: random.Random) -> bool:
        """Decide whether to keep a record given curriculum weights."""
        if self.curriculum_weights is None:
            return True
        weight = self.curriculum_weights.get(category, 0.0)
        if weight <= 0.0:
            return False
        max_weight = max(self.curriculum_weights.values())
        if max_weight <= 0.0:
            return False
        # Accept with probability proportional to weight / max_weight
        return rng.random() < (weight / max_weight)

    def _stream_tokenized(self) -> Iterator[list[int]]:
        """Stream tokenized programs, applying curriculum filtering."""
        rng = random.Random(self.seed)
        for record in _iter_corpus(self.corpus_path):
            category = _extract_category(record)
            if not self._should_keep(category, rng):
                continue
            text = _extract_text(record)
            ids = self._tokenize(text)
            # Skip empty/trivial sequences (just BOS + EOS)
            if len(ids) > 2:
                yield ids

    def _buffered_shuffle(
        self, stream: Iterator[list[int]]
    ) -> Iterator[list[int]]:
        """Buffer-based shuffling: fill buffer, shuffle, yield."""
        if self.shuffle_buffer <= 0:
            yield from stream
            return

        rng = random.Random(self.seed)
        buf: list[list[int]] = []

        for item in stream:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                rng.shuffle(buf)
                yield from buf
                buf.clear()

        # Flush remaining
        if buf:
            rng.shuffle(buf)
            yield from buf

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        """Yield (input_ids, target_ids) batches.

        input_ids:  (B, S) — all tokens except the last
        target_ids: (B, S) — all tokens except the first (shifted by 1)
        """
        stream = self._stream_tokenized()
        shuffled = self._buffered_shuffle(stream)

        # Collect sequences and sort by length for efficient packing
        batch_seqs: list[list[int]] = []

        for ids in shuffled:
            batch_seqs.append(ids)

            if len(batch_seqs) >= self.batch_size:
                # Sort by length to minimize padding within the batch
                batch_seqs.sort(key=len)
                yield self._make_batch(batch_seqs)
                batch_seqs = []

        # Handle the final partial batch
        if batch_seqs:
            batch_seqs.sort(key=len)
            yield self._make_batch(batch_seqs)

    def _make_batch(
        self, seqs: list[list[int]]
    ) -> tuple[Any, Any]:
        """Build (input_ids, target_ids) from a list of token sequences."""
        # Max length in this batch (cap at max_seq_len)
        max_len = min(max(len(s) for s in seqs), self.max_seq_len)

        # Pad all sequences to max_len
        padded = _pad_batch(seqs, max_len)  # (B, max_len)

        # input  = tokens[:-1], target = tokens[1:]
        input_ids = _to_array(padded[:, :-1])
        target_ids = _to_array(padded[:, 1:])
        return input_ids, target_ids

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        total = self._count_programs()
        if total == 0:
            return 0
        return max(1, (total + self.batch_size - 1) // self.batch_size)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the loaded tokenizer."""
        return self.tokenizer.vocab_size()
