"""Tests for scripts/build_prompt_pack.py and the committed prompt pack (story 128.2).

The unit tests need nothing outside this repo. The integration test rebuilds
the whole pack from the sibling toke / toke-corpus checkouts, gates it with
the 131.103 tkc pin, and requires the committed files to equal that build;
it skips, naming what is missing, when the pin or the checkouts are absent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_prompt_pack as bp

PROMPTS = REPO / "prompts"
CFG = json.loads((PROMPTS / "pack.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((PROMPTS / "MANIFEST.json").read_text(encoding="utf-8"))

FAKE_CARD = """# toke syntax card — tkc 9.9.9 / v0.4 idiom

Intro line.

## Loops
```
lp(let i=0;i<n;i=i+1){...};   the only loop
```
Prose that tier a drops.

## Types
- Scalars: `i64 str`

## Patterns — measured canonical forms (catalogue abc; 1 entries)
- acc: sum [acc-sum]
"""
FAKE_MODES = {"Loops": "fences", "Types": "full", "Patterns — measured": "drop"}


def _inputs(card: str = FAKE_CARD) -> bp.Inputs:
    ex = bp.Exemplar(id="X-1", title="One", patterns=[], source="m=x;\nf=main():i64{<0};\n",
                     source_sha256="", record="corpus/x.json", stdout="1\n")
    rc = bp.RepairCase(code="E2002", hint="Use ==.", broken="m=b;\n", fixed="m=f;\n")
    cfg = {"tiers": {"a": {"sections": FAKE_MODES}}, "repair": {"sample": "E2002"}}
    return bp.Inputs(card=card, snippet="", catalogue="{}", cfg=cfg, exemplars=[ex],
                     repairs=[rc], sample_diagnostic='{"error_code":"E2002"}')


# ------------------------------------------------------------------ builder

def test_tier_a_keeps_fences_full_sections_and_drops_the_rest() -> None:
    a = bp.build_tier_a(FAKE_CARD, FAKE_MODES)
    assert "lp(let i=0;i<n;i=i+1)" in a
    assert "Prose that tier a drops." not in a
    assert "- Scalars: `i64 str`" in a
    assert "Patterns" not in a and "Intro line." in a


def test_every_card_section_must_be_named_once() -> None:
    with pytest.raises(bp.PackError, match="matches 0"):
        bp.build_tier_a(FAKE_CARD + "\n## New section\ntext\n", FAKE_MODES)


def test_fences_mode_on_a_section_without_a_fence_is_refused() -> None:
    with pytest.raises(bp.PackError, match="has no fence"):
        bp.build_tier_a(FAKE_CARD, {**FAKE_MODES, "Types": "fences"})


def test_tiers_b_c_d_start_with_the_card_verbatim() -> None:
    built = bp.build_tiers(_inputs())
    assert built.tiers["b"] == FAKE_CARD
    for t in "cd":
        assert built.tiers[t].startswith(FAKE_CARD)
    assert "## Exemplar programs" in built.tiers["c"]
    assert "## Diagnostics and repair" in built.tiers["d"]
    assert '{"error_code":"E2002"}' in built.tiers["d"]


def test_a_card_change_changes_every_tier() -> None:
    old = bp.build_tiers(_inputs()).tiers
    new = bp.build_tiers(_inputs(FAKE_CARD.replace("the only loop", "the one loop"))).tiers
    assert all(old[t] != new[t] for t in "abcd")


def test_patterns_block_must_be_the_catalogue_snippet() -> None:
    snippet = "## Patterns — measured canonical forms (catalogue abc; 1 entries)\n- acc: sum [acc-sum]\n"
    bp.check_patterns_sync(FAKE_CARD, snippet, "abc")
    with pytest.raises(bp.PackError, match="differs"):
        bp.check_patterns_sync(FAKE_CARD, snippet.replace("sum", "total"), "abc")
    with pytest.raises(bp.PackError, match="does not name catalogue"):
        bp.check_patterns_sync(FAKE_CARD, snippet, "def")


def test_an_ungated_fence_is_refused() -> None:
    built = bp.build_tiers(_inputs())
    gated = {"m=x;\nf=main():i64{<0};", "1", "m=b;", "m=f;", '{"error_code":"E2002"}'}
    assert bp.account_fences("c", built.tiers["c"], built.card_part["c"], gated) == 3
    assert bp.account_fences("d", built.tiers["d"], built.card_part["d"], gated) == 4
    with pytest.raises(bp.PackError, match="ungated code block"):
        bp.account_fences("c", built.tiers["c"], built.card_part["c"], gated - {"1"})


@pytest.mark.parametrize("path", ["benchmark/hidden_tests/t1.json",
                                  "toke-eval/benchmark/tasks.jsonl", "x/hidden_tests/y"])
def test_held_out_and_eval_sources_are_refused(path: str) -> None:
    with pytest.raises(bp.PackError, match="refusing"):
        bp.check_source_path(path)


# ------------------------------------------------------- the committed pack

def test_pack_inputs_name_no_held_out_source() -> None:
    for src in CFG["sources"].values():
        bp.check_source_path(src["path"])
    for e in CFG["exemplars"]["programs"]:
        bp.check_source_path(e["record"])


def test_manifest_matches_the_committed_tier_files() -> None:
    assert MANIFEST["pack_version"] == CFG["pack_version"]
    for t, entry in MANIFEST["tiers"].items():
        text = (PROMPTS / entry["file"]).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["sha256"], t
        assert len(text.encode("utf-8")) == entry["bytes"], t
        assert entry["card_gate"]["MISMATCH"] == 0, t
    # tier b is the card verbatim, so its sha is the card_sha
    assert MANIFEST["tiers"]["b"]["sha256"][:12] == MANIFEST["inputs"]["card_sha"]


def test_vendored_exemplars_hash_to_their_records() -> None:
    for e in CFG["exemplars"]["programs"]:
        src = (PROMPTS / "exemplars" / f"{e['id']}.tk").read_text(encoding="utf-8")
        assert hashlib.sha256(src.encode("utf-8")).hexdigest() == e["source_sha256"], e["id"]
    fams = {e["id"].rsplit("v", 1)[0] for e in CFG["exemplars"]["programs"]}
    assert set(MANIFEST["bench128_exclude_families"]) == fams


def test_one_repair_case_per_card_code() -> None:
    card = (PROMPTS / "tier-b-full.md").read_text(encoding="utf-8")
    card_codes = set(re.findall(r"E\d{4}", card))
    cases = json.loads((PROMPTS / "repair_cases.json").read_text(encoding="utf-8"))["cases"]
    codes = [c["code"] for c in cases]
    assert len(codes) == len(set(codes))
    assert card_codes <= set(codes), card_codes - set(codes)


def _pin() -> Path:
    return Path(os.environ.get("TOKE_TKC_PIN") or bp.WORKSPACE / CFG["tkc_pin"]["path"])


def test_committed_pack_equals_a_fresh_gated_build() -> None:
    missing = [str(p) for p in (_pin(), bp.WORKSPACE / "toke" / ".git",
                                bp.WORKSPACE / "toke-corpus" / "regen" / "check_card.py")
               if not p.exists()]
    if missing:
        pytest.skip(f"needs the 131.103 pin and the sibling checkouts: missing {missing}")
    assert bp.main(["--check", "--tkc", str(_pin())]) == 0


def test_gates_reject_a_wrong_output_and_a_wrong_code(tmp_path: Path) -> None:
    if not _pin().exists():
        pytest.skip(f"needs the 131.103 pin at {_pin()}")
    src = 'm=x;\ni=io:std.io;\nf=main():i64{\n  io.println("hi");\n  <0\n};\n'
    good = bp.Exemplar(id="X-1", title="t", patterns=[], source=src,
                       source_sha256=hashlib.sha256(src.encode()).hexdigest(),
                       record="corpus/x.json", stdout="hi\n")
    bp.gate_exemplar(good, _pin(), tmp_path)
    with pytest.raises(bp.PackError, match="stdout"):
        bp.gate_exemplar(bp.Exemplar(**{**good.__dict__, "stdout": "ho\n"}), _pin(), tmp_path)
    with pytest.raises(bp.PackError, match="source_sha256"):
        bp.gate_exemplar(bp.Exemplar(**{**good.__dict__, "source_sha256": "0" * 64}), _pin(), tmp_path)
    broken = "m=x;\nf=main():i64{\n  let n=3;\n  if(n=3){<0};\n  <1\n};\n"
    fixed = broken.replace("n=3){", "n==3){")
    bp.gate_repair(bp.RepairCase(code="E2002", hint="", broken=broken, fixed=fixed), _pin(), tmp_path, 0)
    with pytest.raises(bp.PackError, match="draws"):
        bp.gate_repair(bp.RepairCase(code="E4031", hint="", broken=broken, fixed=fixed),
                       _pin(), tmp_path, 1)
    with pytest.raises(bp.PackError, match="does not check clean"):
        bp.gate_repair(bp.RepairCase(code="E2002", hint="", broken=broken, fixed=broken),
                       _pin(), tmp_path, 2)
