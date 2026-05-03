"""Train a purpose-built BPE tokenizer for the toke language.

Uses the Hugging Face ``tokenizers`` library to train a byte-level BPE
tokenizer on the filtered toke corpus.

Usage:
    python -m tokenizer.train_bpe
    python -m tokenizer.train_bpe --vocab-size 8192 --corpus corpus/filtered/filtered.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths relative to repo root
DEFAULT_CORPUS = "corpus/filtered/filtered.jsonl"
DEFAULT_OUTPUT = "tokenizer/vocab/"
DEFAULT_VOCAB_SIZE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BPE tokenizer for toke")
    p.add_argument("--corpus", type=str, default=DEFAULT_CORPUS,
                   help="Path to JSONL corpus file")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help="Directory to save tokenizer files")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                   help="Target vocabulary size")
    return p.parse_args()


def corpus_iterator(corpus_path: Path):
    """Yield text content from each line of the JSONL corpus.

    TODO:
      - Parse each line as JSON
      - Extract the text/code field
      - Yield as plain string for the tokenizer trainer
    """
    raise NotImplementedError("Corpus iterator not yet implemented")


def train_tokenizer(corpus_path: Path, vocab_size: int) -> object:
    """Train a byte-level BPE tokenizer.

    TODO:
      - from tokenizers import Tokenizer
      - from tokenizers.models import BPE
      - from tokenizers.trainers import BpeTrainer
      - from tokenizers.pre_tokenizers import ByteLevel
      - Configure special tokens: <pad>, <eos>, <unk>
      - Train on corpus via corpus_iterator()
      - Return trained Tokenizer object
    """
    raise NotImplementedError("Tokenizer training not yet implemented")


def save_tokenizer(tokenizer, output_dir: Path) -> None:
    """Save trained tokenizer to disk.

    TODO:
      - output_dir.mkdir(parents=True, exist_ok=True)
      - tokenizer.save(str(output_dir / "tokenizer.json"))
      - Also export vocab.json and merges.txt for inspection
      - Log vocab size and output path
    """
    raise NotImplementedError("Tokenizer saving not yet implemented")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output)
    vocab_size = args.vocab_size

    logger.info("Training BPE tokenizer (vocab_size=%d) on %s", vocab_size, corpus_path)

    # TODO: Uncomment when implemented:
    # tokenizer = train_tokenizer(corpus_path, vocab_size)
    # save_tokenizer(tokenizer, output_dir)
    # logger.info("Tokenizer saved to %s", output_dir)

    logger.info("train_bpe.py is a stub — implementation pending")


if __name__ == "__main__":
    main()
