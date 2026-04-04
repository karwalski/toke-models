"""Tests for the BPE training wrapper (train.py)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_default_config(self, tmp_path: Path) -> None:
        input_file = tmp_path / "train.txt"
        output_dir = tmp_path / "models"
        config = train.build_config(input_file, output_dir)

        assert config["input"] == str(input_file)
        assert config["model_prefix"] == str(output_dir / "toke")
        assert config["model_type"] == "bpe"
        assert config["vocab_size"] == 8000
        assert config["character_coverage"] == 1.0
        assert config["user_defined_symbols"] == ["<STR>"]
        assert config["pad_id"] == -1
        assert config["unk_id"] == 0
        assert config["bos_id"] == 1
        assert config["eos_id"] == 2

    def test_custom_vocab_size(self, tmp_path: Path) -> None:
        config = train.build_config(
            tmp_path / "train.txt", tmp_path / "out", vocab_size=4000
        )
        assert config["vocab_size"] == 4000

    def test_custom_model_type(self, tmp_path: Path) -> None:
        config = train.build_config(
            tmp_path / "train.txt", tmp_path / "out", model_type="unigram"
        )
        assert config["model_type"] == "unigram"

    def test_small_vocab(self, tmp_path: Path) -> None:
        config = train.build_config(
            tmp_path / "train.txt", tmp_path / "out", vocab_size=100
        )
        assert config["vocab_size"] == 100

    def test_large_vocab(self, tmp_path: Path) -> None:
        config = train.build_config(
            tmp_path / "train.txt", tmp_path / "out", vocab_size=32000
        )
        assert config["vocab_size"] == 32000


# ---------------------------------------------------------------------------
# Vocab stats helpers
# ---------------------------------------------------------------------------

class TestVocabStats:
    def test_special_token_count(self) -> None:
        # unk + bos + eos + <STR> = 4
        assert train.count_special_tokens() == 4

    def test_print_vocab_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        train.print_vocab_stats(8000)
        out = capsys.readouterr().out
        assert "8000" in out
        assert "4" in out
        assert "7996" in out


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_missing_input(self, tmp_path: Path) -> None:
        """Missing input file returns exit code 1."""
        rc = train.main([
            "--input", str(tmp_path / "nonexistent.txt"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert rc == 1

    def test_input_is_directory(self, tmp_path: Path) -> None:
        """Input path that is a directory returns exit code 1."""
        rc = train.main([
            "--input", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert rc == 1

    def test_negative_vocab_size(self, tmp_path: Path) -> None:
        """Negative vocab size returns exit code 1."""
        input_file = tmp_path / "train.txt"
        input_file.write_text("hello world\n")
        rc = train.main([
            "--input", str(input_file),
            "--output-dir", str(tmp_path / "out"),
            "--vocab-size", "-1",
        ])
        assert rc == 1

    def test_required_args_missing(self) -> None:
        """Missing required args triggers SystemExit from argparse."""
        with pytest.raises(SystemExit):
            train.main([])

    def test_invalid_model_type(self) -> None:
        """Invalid model-type triggers SystemExit from argparse choices."""
        with pytest.raises(SystemExit):
            train.main([
                "--input", "dummy.txt",
                "--output-dir", "dummy/",
                "--model-type", "invalid",
            ])


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry-run validates input, prints config, and returns 0 without training."""
        input_file = tmp_path / "train.txt"
        input_file.write_text("fn main() { }\n")
        output_dir = tmp_path / "models"

        rc = train.main([
            "--input", str(input_file),
            "--output-dir", str(output_dir),
            "--dry-run",
        ])
        assert rc == 0

        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "Training configuration:" in out
        assert "bpe" in out
        assert "8000" in out
        # Output dir should NOT be created in dry-run.
        assert not output_dir.exists()

    def test_dry_run_custom_vocab(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        input_file = tmp_path / "train.txt"
        input_file.write_text("let x = 1;\n")

        rc = train.main([
            "--input", str(input_file),
            "--output-dir", str(tmp_path / "out"),
            "--vocab-size", "4000",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "4000" in out

    def test_dry_run_missing_input(self, tmp_path: Path) -> None:
        """Dry-run still validates that input exists."""
        rc = train.main([
            "--input", str(tmp_path / "missing.txt"),
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# Training (mocked SentencePiece)
# ---------------------------------------------------------------------------

class TestTrainFunction:
    def test_train_missing_sentencepiece(self) -> None:
        """If sentencepiece is not importable, train() returns 1."""
        config = train.build_config(Path("dummy.txt"), Path("dummy/"))
        with patch.dict("sys.modules", {"sentencepiece": None}):
            rc = train.train(config)
        assert rc == 1

    def test_train_calls_sentencepiece(self, tmp_path: Path) -> None:
        """train() passes config to SentencePieceTrainer.train()."""
        config = train.build_config(tmp_path / "in.txt", tmp_path / "out")

        mock_spm = type(sys)("sentencepiece")
        mock_trainer = type("Trainer", (), {"train": staticmethod(lambda **kw: None)})
        mock_spm.SentencePieceTrainer = mock_trainer

        with patch.dict("sys.modules", {"sentencepiece": mock_spm}):
            rc = train.train(config)
        assert rc == 0

    def test_train_handles_exception(self, tmp_path: Path) -> None:
        """train() catches exceptions from SentencePiece and returns 1."""
        config = train.build_config(tmp_path / "in.txt", tmp_path / "out")

        def raise_error(**kw: object) -> None:
            raise RuntimeError("training exploded")

        mock_spm = type(sys)("sentencepiece")
        mock_trainer = type("Trainer", (), {"train": staticmethod(raise_error)})
        mock_spm.SentencePieceTrainer = mock_trainer

        with patch.dict("sys.modules", {"sentencepiece": mock_spm}):
            rc = train.train(config)
        assert rc == 1
