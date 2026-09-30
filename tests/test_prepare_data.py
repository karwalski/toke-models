"""Tests for finetune/prepare_data.py (story 128.21: this file used to be 0 bytes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import this checkout's finetune/prepare_data.py, whichever clone or worktree runs the test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))

import prepare_data as pd

ONE_PARAM = "m=neg;\nf=neg(a:i64):i64{<0-a};"
TWO_PARAM = "m=add;\nf=add(a:i64;b:i64):i64{<a+b};"


def test_prompt_names_category_and_single_param_signature() -> None:
    prompt = pd.task_id_to_prompt("A-MTH-0001v3", ONE_PARAM)
    assert "`neg(a:i64):i64`" in prompt
    assert "mathematical computation" in prompt


def test_prompt_falls_back_for_unknown_category_and_no_function() -> None:
    prompt = pd.task_id_to_prompt("Z-ZZZ-0001", "m=x;")
    assert prompt.startswith("Write a toke program for general programming.")


@pytest.mark.xfail(strict=True, reason=(
    "defect found by 128.21: the signature is found by splitting the source on ';', "
    "which is also toke's parameter separator, so any multi-parameter function loses "
    "its signature and gets the generic prompt"))
def test_prompt_keeps_multi_param_signature() -> None:
    assert "`add(a:i64;b:i64):i64`" in pd.task_id_to_prompt("A-MTH-0002v1", TWO_PARAM)


def test_direct_example_pairs_prompt_with_source_verbatim() -> None:
    ex = pd.make_direct_example({"task_id": "A-STR-0001v1", "tk_source": ONE_PARAM})
    assert ex["type"] == "direct"
    assert [m["role"] for m in ex["messages"]] == ["user", "assistant"]
    assert ex["messages"][1]["content"] == ONE_PARAM


def test_comparison_example_needs_python_source_and_both_counts() -> None:
    base = {"tk_source": ONE_PARAM, "tk_tokens": 10}
    assert pd.make_comparison_example(base) is None
    assert pd.make_comparison_example(
        {**base, "references": {"python_source": "def f(): pass"}}) is None
    ex = pd.make_comparison_example(
        {**base, "references": {"python_source": "def f(): pass", "python_tokens": 7,
                                "c_source": "int f(){}", "c_tokens": 5}})
    assert ex is not None and ex["type"] == "comparison"
    user, assistant = (m["content"] for m in ex["messages"])
    assert "[PYTHON 7t]" in user and "[C 5t]" in user
    assert assistant == f"[TK 10t]\n{ONE_PARAM}"


def test_comparison_example_makes_no_ratio_claim() -> None:
    # 132.15 removed the "[RATIO] Nx fewer tokens than Python" line; keep it out.
    ex = pd.make_comparison_example(
        {"tk_source": ONE_PARAM, "tk_tokens": 3,
         "references": {"python_source": "x", "python_tokens": 9}})
    assert ex is not None
    text = json.dumps(ex)
    assert "RATIO" not in text and "fewer tokens" not in text


def test_format_chatml_wraps_system_and_every_message() -> None:
    out = pd.format_chatml([{"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "yo"}])
    assert out.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nhi<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\nyo<|im_end|>")
    assert out.count("<|im_start|>") == 3


def test_load_corpus_skips_blank_malformed_and_sourceless(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join([
        json.dumps({"task_id": "a", "tk_source": ONE_PARAM}),
        "",
        "{not json",
        json.dumps({"task_id": "b", "tk_source": ""}),
        json.dumps({"task_id": "c"}),
        json.dumps({"task_id": "d", "tk_source": TWO_PARAM}),
    ]) + "\n")
    assert [e["task_id"] for e in pd.load_corpus(p)] == ["a", "d"]


def test_process_corpus_adds_comparison_only_where_references_exist() -> None:
    entries = [{"task_id": "A-MTH-1", "tk_source": ONE_PARAM},
               {"task_id": "A-MTH-2", "tk_source": ONE_PARAM, "tk_tokens": 4,
                "references": {"python_source": "x", "python_tokens": 8}}]
    assert [e["type"] for e in pd.process_corpus(entries)] == [
        "direct", "direct", "comparison"]


def test_main_writes_a_seeded_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(
        json.dumps({"task_id": f"A-MTH-{i:04d}", "tk_source": ONE_PARAM}) + "\n"
        for i in range(20)))
    out = tmp_path / "out"
    assert pd.main(["--corpus", str(corpus), "--output-dir", str(out),
                    "--split", "0.75", "--seed", "1"]) == 0
    train = (out / "train.jsonl").read_text().splitlines()
    evl = (out / "eval.jsonl").read_text().splitlines()
    assert (len(train), len(evl)) == (15, 5)
    rec = json.loads(train[0])
    assert rec["type"] == "direct" and rec["text"].startswith("<|im_start|>system")

    out2 = tmp_path / "out2"
    pd.main(["--corpus", str(corpus), "--output-dir", str(out2),
             "--split", "0.75", "--seed", "1"])
    assert (out2 / "train.jsonl").read_bytes() == (out / "train.jsonl").read_bytes()


def test_main_rejects_missing_corpus(tmp_path: Path) -> None:
    assert pd.main(["--corpus", str(tmp_path / "absent.jsonl"),
                    "--output-dir", str(tmp_path / "o")]) == 1
