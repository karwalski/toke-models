"""Tests for prepare.py — corpus preparation for BPE tokenizer training."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prepare import (
    deduplicate,
    load_jsonl,
    main,
    replace_string_literals,
    split_data,
    write_text,
)


class TestReplaceStringLiterals:
    def test_simple_string(self) -> None:
        assert replace_string_literals('let s="hello";') == 'let s="<STR>";'

    def test_multiple_strings(self) -> None:
        src = 'f("a";"b")'
        result = replace_string_literals(src)
        assert result == 'f("<STR>";"<STR>")'

    def test_escaped_quote(self) -> None:
        src = r'let s="say \"hi\"";'
        result = replace_string_literals(src)
        assert result == 'let s="<STR>";'

    def test_empty_string(self) -> None:
        assert replace_string_literals('let s="";') == 'let s="<STR>";'

    def test_no_strings(self) -> None:
        src = "M=test;\nF=main():i64{<42};"
        assert replace_string_literals(src) == src

    def test_string_with_newline_escape(self) -> None:
        src = r'let s="line1\nline2";'
        result = replace_string_literals(src)
        assert result == 'let s="<STR>";'


class TestDeduplicate:
    def test_removes_duplicates(self) -> None:
        sources = ["abc", "def", "abc", "ghi", "def"]
        result = deduplicate(sources)
        assert result == ["abc", "def", "ghi"]

    def test_preserves_order(self) -> None:
        sources = ["c", "a", "b", "a"]
        result = deduplicate(sources)
        assert result == ["c", "a", "b"]

    def test_empty_input(self) -> None:
        assert deduplicate([]) == []

    def test_all_unique(self) -> None:
        sources = ["a", "b", "c"]
        assert deduplicate(sources) == sources

    def test_all_identical(self) -> None:
        sources = ["x", "x", "x"]
        assert deduplicate(sources) == ["x"]


class TestLoadJsonl:
    def test_valid_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        entries = [
            {"id": "1", "tk_source": "M=test;"},
            {"id": "2", "tk_source": "M=hello;"},
        ]
        p.write_text("\n".join(json.dumps(e) for e in entries))
        result = load_jsonl(p)
        assert result == ["M=test;", "M=hello;"]

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text('{"tk_source": "ok"}\nnot json\n{"tk_source": "also ok"}\n')
        result = load_jsonl(p)
        assert result == ["ok", "also ok"]

    def test_skips_missing_source(self, tmp_path: Path) -> None:
        p = tmp_path / "nosource.jsonl"
        p.write_text('{"id": "1"}\n{"id": "2", "tk_source": "code"}\n')
        result = load_jsonl(p)
        assert result == ["code"]

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert load_jsonl(p) == []

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.jsonl"
        p.write_text('\n{"tk_source": "a"}\n\n{"tk_source": "b"}\n\n')
        assert load_jsonl(p) == ["a", "b"]


class TestSplitData:
    def test_90_10_split(self) -> None:
        sources = [f"entry{i}" for i in range(100)]
        train, valid = split_data(sources, 0.9)
        assert len(train) == 90
        assert len(valid) == 10

    def test_deterministic(self) -> None:
        sources = [f"entry{i}" for i in range(50)]
        t1, v1 = split_data(sources, 0.9, seed=42)
        t2, v2 = split_data(sources, 0.9, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds(self) -> None:
        sources = [f"entry{i}" for i in range(50)]
        t1, _ = split_data(sources, 0.9, seed=1)
        t2, _ = split_data(sources, 0.9, seed=2)
        assert t1 != t2

    def test_empty_input(self) -> None:
        train, valid = split_data([], 0.9)
        assert train == []
        assert valid == []

    def test_single_entry(self) -> None:
        train, valid = split_data(["only"], 0.9)
        assert len(train) + len(valid) == 1


class TestWriteText:
    def test_writes_sources(self, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        write_text(out, ["M=a;", "M=b;"])
        content = out.read_text()
        assert "M=a;" in content
        assert "M=b;" in content
        assert content.endswith("\n")

    def test_empty_sources(self, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        write_text(out, [])
        assert out.read_text() == "\n"


class TestMainIntegration:
    def test_end_to_end(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus.jsonl"
        entries = [
            {"id": f"e{i}", "tk_source": f'M=mod{i};\nF=f():str{{"hello{i}"}};'}
            for i in range(20)
        ]
        # Add a duplicate
        entries.append(entries[0])
        corpus.write_text("\n".join(json.dumps(e) for e in entries))

        out_dir = tmp_path / "output"
        result = main(["--input", str(corpus), "--output-dir", str(out_dir)])
        assert result == 0
        assert (out_dir / "train.txt").exists()
        assert (out_dir / "valid.txt").exists()

        train_content = (out_dir / "train.txt").read_text()
        valid_content = (out_dir / "valid.txt").read_text()
        # 20 unique entries (1 dup removed), 90/10 split → 18 train, 2 valid
        assert train_content.count("M=mod") == 18
        assert valid_content.count("M=mod") == 2
        # String literals should be replaced
        assert "<STR>" in train_content

    def test_missing_input(self, tmp_path: Path) -> None:
        result = main(["--input", str(tmp_path / "nonexistent.jsonl"), "--output-dir", str(tmp_path)])
        assert result == 1
