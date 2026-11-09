"""Tests for finetune/prepare_data.py (story 128.21: this file used to be 0 bytes)."""

from __future__ import annotations

import importlib.util
import json
import os
import re
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


def test_prompt_keeps_multi_param_signature() -> None:
    # 128.26: this was a strict xfail — splitting the source on ';' (also toke's
    # parameter separator) lost every multi-parameter signature.
    assert "`add(a:i64;b:i64):i64`" in pd.task_id_to_prompt("A-MTH-0002v1", TWO_PARAM)


def test_prompt_keeps_three_param_legacy_signature() -> None:
    src = 'M=isBetween;F=isBetween(x:u64;lo:u64;hi:u64):bool{if(x>hi){<false};<true};'
    assert "`isBetween(x:u64;lo:u64;hi:u64):bool`" in pd.task_id_to_prompt(
        "A-CND-0003v1", src)


@pytest.mark.parametrize("sig", [
    "lookup(m:@(str:i64);k:str):i64!$err",
    "apply(g:(i64;i64):i64;xs:@i64):@i64",
    "count(xs:[[i64]];m:[Str:i64]):i64",
])
def test_extract_signature_handles_nested_types(sig: str) -> None:
    src = f'm=t;\ni=io:std.io;\nf={sig}{{let s="a;b{{";<0}};\nf=other(a:i64):i64{{<a}};'
    assert pd.extract_signature(src) == sig


def test_extract_signature_skips_body_assignments_and_strings() -> None:
    # 'f=' inside a body or a string is not a declaration.
    assert pd.extract_signature('m=x;c="f=no(a:i64):i64{":str;') == ""
    src = 'm=x;f=main():i64{let f=mut.0;f=f+1;<f};'
    assert pd.extract_signature(src) == "main():i64"


def test_extract_signature_skips_bodyless_declaration() -> None:
    src = ("M=p;T=CalcErr{Bad:Str};F=strToF64(s:Str):f64!CalcErr;"
           "F=divide(a:Str;b:Str):f64!CalcErr{<strToF64(a)!CalcErr};")
    assert pd.extract_signature(src) == "divide(a:Str;b:Str):f64!CalcErr"
    assert pd.extract_signature("M=p;F=ext(s:Str):Str;") == ""


def test_extract_signature_rejects_malformed_declaration() -> None:
    assert pd.extract_signature("m=x;f=broken(a:i64;") == ""
    assert pd.extract_signature("m=x;f=(a:i64):i64{<a};") == ""


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
    assert "[PYTHON]\ndef f(): pass" in user and "[C]\nint f(){}" in user
    assert assistant == f"[TK]\n{ONE_PARAM}"


def test_comparison_example_makes_no_ratio_claim() -> None:
    # 132.15 removed the "[RATIO] Nx fewer tokens than Python" line; keep it out.
    ex = pd.make_comparison_example(
        {"tk_source": ONE_PARAM, "tk_tokens": 3,
         "references": {"python_source": "x", "python_tokens": 9}})
    assert ex is not None
    text = json.dumps(ex)
    assert "RATIO" not in text and "fewer tokens" not in text


# The marker shape toke-corpus scripts/export_guards.py COUNT_MARKER blocks (131.93).
_COUNT_MARKER = re.compile(
    r"\[(PYTHON|TK|TOKE|C|CPP|JAVA|JS|JAVASCRIPT|TS|TYPESCRIPT|RUST|GO)\s+\d+t")


def test_comparison_example_carries_no_unsourced_token_counts() -> None:
    # 131.97: no tokenizer is named for python_tokens / c_tokens / tk_tokens, so
    # the per-side "[LANG Nt]" counts are dropped rather than published.
    ex = pd.make_comparison_example(
        {"tk_source": ONE_PARAM, "tk_tokens": 3,
         "references": {"python_source": "x", "python_tokens": 9,
                        "c_source": "int x;", "c_tokens": 4}})
    assert ex is not None
    text = json.dumps(ex)
    assert not _COUNT_MARKER.search(text)
    assert "9" not in text and "3t" not in text and "4t" not in text


def _export_guards():
    """toke-corpus's export_guards module, if a sibling checkout is available."""
    root = os.environ.get("TOKE_CORPUS_DIR") or str(
        Path(__file__).resolve().parents[2] / "toke-corpus")
    path = Path(root) / "scripts" / "export_guards.py"
    if not path.is_file():
        pytest.skip(f"toke-corpus export_guards.py not found at {path} (set TOKE_CORPUS_DIR)")
    spec = importlib.util.spec_from_file_location("export_guards", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exporter_output_passes_toke_corpus_export_guards(tmp_path: Path) -> None:
    # 131.97: run the real 131.63/131.93 guard over this exporter's output.
    guards = _export_guards()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({
        "task_id": f"A-MTH-{i:04d}", "tk_source": ONE_PARAM, "tk_tokens": 5 + i,
        "references": {"python_source": "def neg(a): return -a", "python_tokens": 9 + i,
                       "c_source": "long neg(long a){return -a;}", "c_tokens": 12 + i},
    }) + "\n" for i in range(6)))
    out = tmp_path / "out"
    assert pd.main(["--corpus", str(corpus), "--output-dir", str(out), "--split", "0.5"]) == 0
    for name in ("train.jsonl", "eval.jsonl"):
        n, bad = guards.scan_jsonl(out / name)
        assert n > 0 and bad == []


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


def test_load_corpus_excludes_the_retired_w1010_bifi_record(tmp_path: Path) -> None:
    # 135.34 / owner decision §4.5: the legacy BIFI record carrying tkc 0.1.0's
    # retired W1010 fix string never reaches training prep.
    bad = "BIFI-missing_semicolon-21952-7371771c"
    assert bad in pd.EXCLUDED_RECORD_IDS
    p = tmp_path / "c.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in [
        {"id": bad, "task_id": "BIFI-missing_semicolon", "tk_source": ONE_PARAM},
        {"id": "BIFI-missing_semicolon-1-aaaaaaaa", "task_id": "BIFI-missing_semicolon",
         "tk_source": TWO_PARAM},
    ]))
    assert [e["id"] for e in pd.load_corpus(p)] == ["BIFI-missing_semicolon-1-aaaaaaaa"]


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
