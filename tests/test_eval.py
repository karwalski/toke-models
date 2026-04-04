"""Tests for eval.py — tokenizer evaluation against cl100k_base."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from eval import (
    build_report,
    compute_compression_ratio,
    compute_fertility,
    compute_token_stats,
    compute_vocab_utilization,
    format_summary,
    load_programs,
    main,
)


class TestLoadPrograms:
    def test_double_newline_split(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.txt"
        p.write_text("M=a;\nF=f():i64{<1};\n\nM=b;\nF=g():i64{<2};\n")
        result = load_programs(p)
        assert len(result) == 2
        assert result[0] == "M=a;\nF=f():i64{<1};"
        assert result[1] == "M=b;\nF=g():i64{<2};"

    def test_single_program(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.txt"
        p.write_text("M=only;\n")
        result = load_programs(p)
        assert result == ["M=only;"]

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.txt"
        p.write_text("")
        assert load_programs(p) == []

    def test_whitespace_only(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.txt"
        p.write_text("   \n\n  \n")
        assert load_programs(p) == []

    def test_strips_trailing_whitespace(self, tmp_path: Path) -> None:
        p = tmp_path / "valid.txt"
        p.write_text("M=a;  \n\n\nM=b;\n")
        result = load_programs(p)
        assert result[0] == "M=a;"


class TestComputeTokenStats:
    def test_basic_stats(self) -> None:
        # 5 programs with known token counts: 10, 20, 30, 40, 50
        token_lists = [list(range(n)) for n in [10, 20, 30, 40, 50]]
        stats = compute_token_stats(token_lists)
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        # p95 index = int(5 * 0.95) = 4 -> sorted[4] = 50
        assert stats["p95"] == 50.0

    def test_single_program(self) -> None:
        stats = compute_token_stats([[1, 2, 3]])
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["p95"] == 3.0

    def test_empty_input(self) -> None:
        stats = compute_token_stats([])
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["p95"] == 0.0


class TestComputeCompressionRatio:
    def test_equal_tokens(self) -> None:
        toke = [[1, 2, 3], [4, 5, 6]]
        baseline = [[1, 2, 3], [4, 5, 6]]
        assert compute_compression_ratio(toke, baseline) == 1.0

    def test_fewer_tokens(self) -> None:
        toke = [[1, 2]]       # 2 total
        baseline = [[1, 2, 3, 4]]  # 4 total
        assert compute_compression_ratio(toke, baseline) == 0.5

    def test_more_tokens(self) -> None:
        toke = [[1, 2, 3, 4]]  # 4 total
        baseline = [[1, 2]]     # 2 total
        assert compute_compression_ratio(toke, baseline) == 2.0

    def test_empty_baseline(self) -> None:
        assert compute_compression_ratio([[1]], []) == 0.0

    def test_both_empty(self) -> None:
        assert compute_compression_ratio([], []) == 0.0


class TestComputeVocabUtilization:
    def test_full_utilization(self) -> None:
        tokens = [[0, 1], [2, 3]]
        assert compute_vocab_utilization(tokens, 4) == 1.0

    def test_partial_utilization(self) -> None:
        tokens = [[0, 1], [0, 1]]
        assert compute_vocab_utilization(tokens, 4) == 0.5

    def test_empty_tokens(self) -> None:
        assert compute_vocab_utilization([], 100) == 0.0

    def test_zero_vocab(self) -> None:
        assert compute_vocab_utilization([[1]], 0) == 0.0


class TestComputeFertility:
    def test_basic_fertility(self) -> None:
        programs = ["abcd"]  # 4 chars
        tokens = [[1, 2]]    # 2 tokens
        assert compute_fertility(programs, tokens) == 0.5

    def test_multiple_programs(self) -> None:
        programs = ["ab", "abcd"]  # 2 chars, 4 chars
        tokens = [[1], [1, 2]]     # 1 tok, 2 toks -> 0.5, 0.5
        assert compute_fertility(programs, tokens) == 0.5

    def test_empty_programs(self) -> None:
        assert compute_fertility([], []) == 0.0

    def test_empty_string_program(self) -> None:
        # A program with empty string should be skipped
        programs = ["", "abcd"]
        tokens = [[], [1, 2]]
        result = compute_fertility(programs, tokens)
        assert result == 0.5


class TestBuildReport:
    def test_report_structure(self) -> None:
        programs = ["abc", "defgh"]
        toke_tokens = [[1, 2], [3, 4, 5]]
        baseline_tokens = [[10, 20, 30], [40, 50, 60, 70]]
        report = build_report(programs, toke_tokens, baseline_tokens, vocab_size=1000)
        assert report["program_count"] == 2
        assert "mean" in report["toke"]
        assert "median" in report["toke"]
        assert "p95" in report["toke"]
        assert "mean" in report["baseline"]
        assert "compression_ratio" in report
        assert "vocab_utilization" in report
        assert "fertility" in report
        assert report["vocab_size"] == 1000

    def test_report_values(self) -> None:
        programs = ["abcdef"]  # 6 chars
        toke_tokens = [[1, 2, 3]]  # 3 tokens
        baseline_tokens = [[10, 20, 30, 40, 50, 60]]  # 6 tokens
        report = build_report(programs, toke_tokens, baseline_tokens, vocab_size=100)
        assert report["compression_ratio"] == 0.5
        assert report["fertility"] == 0.5
        assert report["vocab_utilization"] == 3 / 100


class TestFormatSummary:
    def test_contains_key_fields(self) -> None:
        report = {
            "program_count": 10,
            "vocab_size": 8000,
            "toke": {"mean": 25.0, "median": 22.0, "p95": 50.0},
            "baseline": {"mean": 35.0, "median": 30.0, "p95": 70.0},
            "compression_ratio": 0.7143,
            "vocab_utilization": 0.45,
            "fertility": 0.32,
        }
        summary = format_summary(report)
        assert "10" in summary
        assert "8000" in summary
        assert "0.7143" in summary
        assert "0.4500" in summary
        assert "0.3200" in summary
        assert "toke" in summary.lower()
        assert "baseline" in summary.lower()


class TestDryRun:
    def test_dry_run_succeeds(self, tmp_path: Path) -> None:
        model = tmp_path / "toke.model"
        model.write_text("fake model")
        data = tmp_path / "valid.txt"
        data.write_text("M=a;\n\nM=b;\n")

        result = main([
            "--model", str(model),
            "--test-data", str(data),
            "--dry-run",
        ])
        assert result == 0

    def test_dry_run_missing_model(self, tmp_path: Path) -> None:
        data = tmp_path / "valid.txt"
        data.write_text("M=a;\n")

        result = main([
            "--model", str(tmp_path / "missing.model"),
            "--test-data", str(data),
            "--dry-run",
        ])
        assert result == 1

    def test_dry_run_missing_data(self, tmp_path: Path) -> None:
        model = tmp_path / "toke.model"
        model.write_text("fake model")

        result = main([
            "--model", str(model),
            "--test-data", str(tmp_path / "missing.txt"),
            "--dry-run",
        ])
        assert result == 1

    def test_dry_run_empty_data(self, tmp_path: Path) -> None:
        model = tmp_path / "toke.model"
        model.write_text("fake model")
        data = tmp_path / "valid.txt"
        data.write_text("")

        result = main([
            "--model", str(model),
            "--test-data", str(data),
            "--dry-run",
        ])
        assert result == 1


class TestMainWithMocks:
    def test_full_evaluation(self, tmp_path: Path) -> None:
        model = tmp_path / "toke.model"
        model.write_text("fake model")
        data = tmp_path / "valid.txt"
        data.write_text("M=a;\nF=f():i64{<1};\n\nM=b;\nF=g():i64{<2};\n")
        output = tmp_path / "report.json"

        mock_sp_instance = MagicMock()
        mock_sp_instance.encode.side_effect = lambda text, out_type=None: list(range(len(text) // 3))
        mock_sp_instance.GetPieceSize.return_value = 8000

        mock_spm = MagicMock()
        mock_spm.SentencePieceProcessor.return_value = mock_sp_instance

        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda text: list(range(len(text) // 2))

        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc

        with patch.dict("sys.modules", {
            "sentencepiece": mock_spm,
            "tiktoken": mock_tiktoken,
        }):
            result = main([
                "--model", str(model),
                "--test-data", str(data),
                "--output", str(output),
            ])

        assert result == 0
        assert output.exists()
        report = json.loads(output.read_text())
        assert report["program_count"] == 2
        assert "toke" in report
        assert "baseline" in report
        assert "compression_ratio" in report
        assert "vocab_utilization" in report
        assert "fertility" in report

    def test_json_report_is_valid(self, tmp_path: Path) -> None:
        model = tmp_path / "toke.model"
        model.write_text("fake model")
        data = tmp_path / "valid.txt"
        data.write_text("M=test;\n")
        output = tmp_path / "report.json"

        mock_sp_instance = MagicMock()
        mock_sp_instance.encode.return_value = [1, 2, 3]
        mock_sp_instance.GetPieceSize.return_value = 500

        mock_spm = MagicMock()
        mock_spm.SentencePieceProcessor.return_value = mock_sp_instance

        mock_enc = MagicMock()
        mock_enc.encode.return_value = [10, 20, 30, 40]

        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc

        with patch.dict("sys.modules", {
            "sentencepiece": mock_spm,
            "tiktoken": mock_tiktoken,
        }):
            result = main([
                "--model", str(model),
                "--test-data", str(data),
                "--output", str(output),
            ])

        assert result == 0
        report = json.loads(output.read_text())
        # Single program with 3 toke tokens, 4 baseline tokens
        assert report["compression_ratio"] == 0.75
        assert report["toke"]["mean"] == 3.0
        assert report["baseline"]["mean"] == 4.0

    def test_no_output_flag(self, tmp_path: Path) -> None:
        """Without --output, should print to stdout only and succeed."""
        model = tmp_path / "toke.model"
        model.write_text("fake model")
        data = tmp_path / "valid.txt"
        data.write_text("M=a;\n")

        mock_sp_instance = MagicMock()
        mock_sp_instance.encode.return_value = [1, 2]
        mock_sp_instance.GetPieceSize.return_value = 100

        mock_spm = MagicMock()
        mock_spm.SentencePieceProcessor.return_value = mock_sp_instance

        mock_enc = MagicMock()
        mock_enc.encode.return_value = [10, 20, 30]

        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.return_value = mock_enc

        with patch.dict("sys.modules", {
            "sentencepiece": mock_spm,
            "tiktoken": mock_tiktoken,
        }):
            result = main([
                "--model", str(model),
                "--test-data", str(data),
            ])

        assert result == 0
