#!/usr/bin/env python3
"""Build and gate the graduated "how to write toke" prompt pack (story 128.2).

The pack is four system-prompt tiers derived from toke-corpus's syntax card v2
(``regen/syntax_card.md``) and toke's pattern catalogue
(``patterns/card_snippet.md`` + ``patterns/catalogue.json``):

  a  minimal card          the card's rule sections cut to their code
  b  full card             the card verbatim, Patterns block included
  c  card + exemplars      b + whole programs from the v0.4 corpus
  d  card + repair guide   b + how to read a diagnostic + broken/fixed pairs

Inputs and tier definitions live in ``prompts/pack.json``. The builder writes
``prompts/tier-*.md`` and ``prompts/MANIFEST.json`` (pack version, input
shas, the tkc pin, each tier's sha256, and the gate tallies).

Every fenced code block in every tier is gated with the 131.103 tkc pin:

  * card text      ``toke-corpus/regen/check_card.py`` (the card's own gate,
                   polarity-aware) run on the tier's card part: MISMATCH 0;
  * exemplars      the vendored source must hash to its corpus record's
                   ``source_sha256``; it must build, run to exit 0, print the
                   recorded stdout, and draw no lint diagnostic at all;
  * repair pairs   the broken program must draw exactly its error code first
                   under ``--check``; the fixed one must build, run to exit 0
                   and draw no lint warning or error;
  * the sample diagnostic is the pin's own output on the named repair case.

A final accounting pass re-reads every fence in each finished tier and fails
if one is not covered by one of those gates.

Inputs are read from the sibling checkouts' ``main`` branch (``git show
main:<path>``), so a dirty working tree cannot leak into the pack. The pack
records content shas only, never commit shas: it changes when the card, the
catalogue, an exemplar or a repair case changes, and not otherwise.

Usage::

    python3 scripts/build_prompt_pack.py            # rebuild and write
    python3 scripts/build_prompt_pack.py --check    # rebuild, gate, compare; exit 1 on drift

No model is called; nothing here evaluates a model (that is 128.10).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def find_workspace() -> Path:
    """$TOKE_WORKSPACE, else the nearest ancestor holding toke/ and toke-corpus/
    side by side (a git worktree may sit deeper than the main checkout)."""
    env = os.environ.get("TOKE_WORKSPACE")
    if env:
        return Path(env)
    for d in REPO.parents:
        if (d / "toke").is_dir() and (d / "toke-corpus").is_dir():
            return d
    return REPO.parent


WORKSPACE = find_workspace()
PROMPTS = REPO / "prompts"
PACK_JSON = PROMPTS / "pack.json"

# Never take prompt material from a held-out set or the eval repo (AGENTS.md section 5).
FORBIDDEN = re.compile(r"hidden_tests|toke-eval|benchmark/")
LEGACY = re.compile(r"\$(?:i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|bool|str|void)\b")
STRING_LIT = re.compile(r'"(?:\\.|[^"\\])*"')
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
SUMMARY = re.compile(r"MISMATCH (\d+)")


class PackError(Exception):
    """An input or a gate failed; the message says which."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(digest: str) -> str:
    return digest[:12]


# --------------------------------------------------------------------- inputs

def check_source_path(path: str) -> None:
    if FORBIDDEN.search(path):
        raise PackError(f"refusing prompt material from {path!r} (held-out or eval tree)")


def read_from_main(root: Path, path: str) -> str:
    """The file's text on the checkout's ``main`` branch."""
    check_source_path(path)
    r = subprocess.run(["git", "-C", str(root), "show", f"main:{path}"],
                       capture_output=True, check=False, text=True)
    if r.returncode != 0:
        raise PackError(f"cannot read main:{path} in {root}: {r.stderr.strip()}")
    return r.stdout


def split_sections(card: str) -> tuple[str, list[tuple[str, str]]]:
    """(preamble, [(heading, body)]) split on ``## `` headings outside fences."""
    pre: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in card.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
            continue
        (sections[-1][1] if sections else pre).append(line)
    return "".join(pre), [(h, "".join(b)) for h, b in sections]


def fences_only(body: str) -> str:
    """A section's heading line and its fenced blocks, nothing else."""
    heading = body.splitlines(keepends=True)[0]
    blocks = [m.group(0) for m in FENCE.finditer(body)]
    if not blocks:
        raise PackError(f"section {heading.strip()!r} has no fence; give it 'full' or 'drop'")
    return heading + "".join(b if b.endswith("\n") else b + "\n" for b in blocks) + "\n"


def mode_for(heading: str, modes: dict[str, str]) -> str:
    hits = [m for key, m in modes.items() if heading.startswith(key)]
    if len(hits) != 1:
        raise PackError(f"card section {heading!r} matches {len(hits)} tier-a entries; "
                        "name every card section exactly once in pack.json")
    return hits[0]


def check_patterns_sync(card: str, snippet: str, catalogue_sha: str) -> None:
    """The card's Patterns block must be the catalogue's own snippet, for this catalogue."""
    if snippet.strip() not in card:
        raise PackError("the card's Patterns block differs from patterns/card_snippet.md: "
                        "regenerate the card from the catalogue first")
    head = snippet.splitlines()[0] if snippet else ""
    if f"catalogue {catalogue_sha}" not in head:
        raise PackError(f"card_snippet.md header {head!r} does not name catalogue {catalogue_sha}")


# ---------------------------------------------------------------------- tiers

@dataclass
class Exemplar:
    id: str
    title: str
    patterns: list[str]
    source: str
    source_sha256: str
    record: str
    stdout: str


@dataclass
class RepairCase:
    code: str
    hint: str
    broken: str
    fixed: str


@dataclass
class Inputs:
    card: str
    snippet: str
    catalogue: str
    cfg: dict[str, Any]
    exemplars: list[Exemplar]
    repairs: list[RepairCase]
    sample_diagnostic: str = ""


@dataclass
class Built:
    tiers: dict[str, str]
    card_part: dict[str, str]
    gated_blocks: dict[str, set[str]] = field(default_factory=dict)


def fence(code: str, lang: str = "") -> str:
    return f"```{lang}\n{code.rstrip(chr(10))}\n```\n"


def build_tier_a(card: str, modes: dict[str, str]) -> str:
    pre, sections = split_sections(card)
    out = [pre]
    for heading, body in sections:
        mode = mode_for(heading, modes)
        if mode == "full":
            out.append(body)
        elif mode == "fences":
            out.append(fences_only(body))
        elif mode != "drop":
            raise PackError(f"unknown tier-a mode {mode!r} for {heading!r}")
    return "".join(out).rstrip("\n") + "\n"


def exemplar_section(exemplars: list[Exemplar]) -> str:
    parts = ["\n## Exemplar programs\n",
             ("Whole programs that pass `tkc --check`, build, run and print exactly the output\n"
              "shown. Write in the same style.\n")]
    for e in exemplars:
        parts.append(f"\n### {e.title}\n")
        parts.append(fence(e.source))
        parts.append("Output:\n")
        parts.append(fence(e.stdout, "text"))
    return "".join(parts)


def repair_section(repairs: list[RepairCase], sample_diagnostic: str,
                   errors_titles: dict[str, str], sample_code: str) -> str:
    parts = ["\n## Diagnostics and repair\n",
             ("`tkc --check` reports every error as one JSON object per line. The fields that\n"
              "matter for a repair: `error_code` (stable, never renumbered), `message`, `pos`\n"
              "(`line`, `col`), `source_line`, and `fix` when the compiler knows the one correct\n"
              f"edit. The record for the {sample_code} program below:\n"),
             fence(sample_diagnostic, "json"),
             ("Repair rules:\n"
              "- Fix the first diagnostic first; later ones are often its consequences.\n"
              "- When a record carries `fix`, apply it exactly. When it does not, change only the\n"
              "  construct at `pos` and keep the function signatures.\n"
              "- Re-check after every edit; stop after 3 rounds and report what still fails.\n"),
             ]
    for c in repairs:
        title = errors_titles.get(c.code, "")
        parts.append(f"\n### {c.code}{' — ' + title if title else ''}\n")
        parts.append(c.hint.rstrip() + "\n")
        parts.append("Broken:\n")
        parts.append(fence(c.broken))
        parts.append("Fixed:\n")
        parts.append(fence(c.fixed))
    return "".join(parts)


def build_tiers(inp: Inputs, errors_titles: dict[str, str] | None = None) -> Built:
    """Assemble the four tier texts. Pure: no compiler, no git."""
    full = inp.card.rstrip("\n") + "\n"
    tier_a = build_tier_a(inp.card, inp.cfg["tiers"]["a"]["sections"])
    tiers = {
        "a": tier_a,
        "b": full,
        "c": full + exemplar_section(inp.exemplars),
        "d": full + repair_section(inp.repairs, inp.sample_diagnostic, errors_titles or {},
                                  inp.cfg["repair"]["sample"]),
    }
    card_part = {"a": tier_a, "b": full, "c": full, "d": full}
    return Built(tiers=tiers, card_part=card_part)


# ---------------------------------------------------------------------- gates

def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, check=False, text=True, timeout=120, **kw)


def check_pin(tkc: Path, pin: dict[str, str]) -> None:
    if not tkc.exists():
        raise PackError(f"tkc pin not found at {tkc}")
    digest = hashlib.sha256(tkc.read_bytes()).hexdigest()
    if digest != pin["sha256"]:
        raise PackError(f"{tkc} sha256 {digest} is not the {pin['story']} pin {pin['sha256']}")
    ver = _run([str(tkc), "--version"]).stdout.strip()
    if ver != pin["version"]:
        raise PackError(f"{tkc} reports {ver!r}, expected {pin['version']!r}")


def gate_card(text: str, check_card: Path, tkc: Path) -> dict[str, Any]:
    """Run the card's own compile gate on a tier's card part. MISMATCH must be 0."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "card.md"
        p.write_text(text, encoding="utf-8")
        env = dict(os.environ, TOKE_TKC_PIN=str(tkc), TKC=str(tkc))
        r = _run([sys.executable, str(check_card), "--card", str(p)], env=env)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("fragments ")), "")
    m = SUMMARY.search(line)
    if not m:
        raise PackError(f"check_card.py gave no summary (rc {r.returncode}): {r.stdout[-400:]}{r.stderr[-400:]}")
    if int(m.group(1)) != 0 or r.returncode != 0:
        raise PackError(f"card gate failed: {line}\n{r.stdout[-1500:]}")
    return {"MISMATCH": int(m.group(1)), "summary": re.sub(r"\s+", " ", line).strip()}


def check_codes(tkc: Path, path: Path) -> list[str]:
    r = _run([str(tkc), "--check", path.name], cwd=path.parent)
    codes = []
    for ln in (r.stdout + r.stderr).splitlines():
        if ln.startswith("{"):
            codes.append(json.loads(ln).get("error_code") or "")
    return codes


def lint_records(tkc: Path, path: Path) -> list[dict[str, Any]]:
    r = _run([str(tkc), "--lint", path.name], cwd=path.parent)
    return [json.loads(ln) for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("{")]


def build_and_run(tkc: Path, src: str, td: Path, name: str) -> subprocess.CompletedProcess[str]:
    p = td / f"{name}.tk"
    p.write_text(src, encoding="utf-8")
    b = td / f"{name}.bin"
    r = _run([str(tkc), p.name, "--config=/dev/null", "--out", b.name], cwd=td)
    if r.returncode != 0 or not b.exists():
        raise PackError(f"{name}: does not build with the pin: {(r.stdout + r.stderr)[:400]}")
    return _run([str(b)], input="", cwd=td)


def gate_exemplar(e: Exemplar, tkc: Path, td: Path) -> None:
    if sha256(e.source) != e.source_sha256:
        raise PackError(f"{e.id}: vendored source does not hash to its record's source_sha256")
    code = STRING_LIT.sub('""', e.source)
    if re.search(r"[A-Z_]", code) or LEGACY.search(code):
        raise PackError(f"{e.id}: legacy spelling (uppercase, `_` or a `$`-scalar) outside strings")
    p = td / f"{e.id}.tk"
    p.write_text(e.source, encoding="utf-8")
    lints = lint_records(tkc, p)
    if lints:
        raise PackError(f"{e.id}: lint is not clean: {[x.get('rule') for x in lints]}")
    run = build_and_run(tkc, e.source, td, e.id)
    if run.returncode != 0:
        raise PackError(f"{e.id}: exits {run.returncode}")
    if run.stdout != e.stdout:
        raise PackError(f"{e.id}: stdout {run.stdout!r} != recorded {e.stdout!r}")


def gate_repair(c: RepairCase, tkc: Path, td: Path, idx: int) -> None:
    p = td / f"broken{idx}.tk"
    p.write_text(c.broken, encoding="utf-8")
    codes = check_codes(tkc, p)
    if not codes or codes[0] != c.code:
        raise PackError(f"repair {c.code}: the broken program draws {codes or 'no error'}")
    f = td / f"fixed{idx}.tk"
    f.write_text(c.fixed, encoding="utf-8")
    if check_codes(tkc, f):
        raise PackError(f"repair {c.code}: the fixed program does not check clean")
    bad = [x.get("rule") for x in lint_records(tkc, f) if x.get("severity") in ("warning", "error")]
    if bad:
        raise PackError(f"repair {c.code}: the fixed program draws lint {bad}")
    run = build_and_run(tkc, c.fixed, td, f"fixed{idx}")
    if run.returncode != 0:
        raise PackError(f"repair {c.code}: the fixed program exits {run.returncode}")


def sample_diagnostic(tkc: Path, c: RepairCase) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prog.tk"
        p.write_text(c.broken, encoding="utf-8")
        r = _run([str(tkc), "--check", p.name], cwd=td)
    first = next((ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("{")), "")
    if not first:
        raise PackError(f"no diagnostic for the sample case {c.code}")
    return first


def account_fences(tier: str, text: str, card_part: str, gated: set[str]) -> int:
    """Every fence in the tier is in its gated card part or is a gated block."""
    card_blocks = {m.group(1).rstrip("\n") for m in FENCE.finditer(card_part)}
    n = 0
    for m in FENCE.finditer(text):
        body = m.group(1).rstrip("\n")
        n += 1
        if body in card_blocks or body in gated:
            continue
        raise PackError(f"tier {tier}: ungated code block:\n{body[:300]}")
    return n


# ----------------------------------------------------------------- top level

def load_inputs(cfg: dict[str, Any], corpus_root: Path, toke_root: Path) -> Inputs:
    src = cfg["sources"]
    roots = {"toke-corpus": corpus_root, "toke": toke_root}
    card = read_from_main(roots[src["card"]["repo"]], src["card"]["path"])
    snippet = read_from_main(roots[src["card_snippet"]["repo"]], src["card_snippet"]["path"])
    catalogue = read_from_main(roots[src["catalogue"]["repo"]], src["catalogue"]["path"])
    exemplars = []
    for e in cfg["exemplars"]["programs"]:
        check_source_path(e["record"])
        path = PROMPTS / "exemplars" / f"{e['id']}.tk"
        exemplars.append(Exemplar(id=e["id"], title=e["title"], patterns=e["patterns"],
                                  source=path.read_text(encoding="utf-8"),
                                  source_sha256=e["source_sha256"], record=e["record"],
                                  stdout=e["stdout"]))
    rc = json.loads((PROMPTS / cfg["repair"]["cases"]).read_text(encoding="utf-8"))
    repairs = [RepairCase(**c) for c in rc["cases"]]
    return Inputs(card=card, snippet=snippet, catalogue=catalogue, cfg=cfg,
                  exemplars=exemplars, repairs=repairs)


def errors_titles(toke_root: Path) -> dict[str, str]:
    """Each code's title line from toke's docs/reference/errors.md."""
    text = read_from_main(toke_root, "docs/reference/errors.md")
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^### (E\d{4})\s*\n\s*\n\*\*(.+?)\*\*\s*$", text, re.MULTILINE)}


def build(corpus_root: Path, toke_root: Path, tkc: Path) -> tuple[dict[str, str], dict[str, Any]]:
    cfg = json.loads(PACK_JSON.read_text(encoding="utf-8"))
    check_pin(tkc, cfg["tkc_pin"])
    inp = load_inputs(cfg, corpus_root, toke_root)
    catalogue_sha = short(sha256(inp.catalogue))
    check_patterns_sync(inp.card, inp.snippet, catalogue_sha)
    ids = {e["id"] for e in json.loads(inp.catalogue)["entries"]}
    for e in inp.exemplars:
        unknown = [p for p in e.patterns if p not in ids]
        if unknown:
            raise PackError(f"{e.id}: patterns not in the catalogue: {unknown}")
    if not inp.repairs:
        raise PackError("no repair cases")
    sample = next((c for c in inp.repairs if c.code == cfg["repair"]["sample"]), None)
    if sample is None:
        raise PackError(f"repair sample {cfg['repair']['sample']!r} is not a repair case")
    inp.sample_diagnostic = sample_diagnostic(tkc, sample)
    titles = errors_titles(toke_root)
    built = build_tiers(inp, titles)

    check_card = corpus_root / "regen" / "check_card.py"
    gates: dict[str, Any] = {"card": {}, "exemplars": [], "repairs": []}
    gated: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        for e in inp.exemplars:
            gate_exemplar(e, tkc, td)
            gated |= {e.source.rstrip("\n"), e.stdout.rstrip("\n")}
            gates["exemplars"].append(e.id)
        for i, c in enumerate(inp.repairs):
            gate_repair(c, tkc, td, i)
            gated |= {c.broken.rstrip("\n"), c.fixed.rstrip("\n")}
            gates["repairs"].append(c.code)
    gated.add(inp.sample_diagnostic)
    gated_by_tier = {"a": set(), "b": set(), "c": gated, "d": gated}
    card_results: dict[str, dict[str, Any]] = {}
    for t in "abcd":
        part = built.card_part[t]
        key = sha256(part)
        if key not in card_results:
            card_results[key] = gate_card(part, check_card, tkc)
        gates["card"][t] = card_results[key]
        gates.setdefault("fences", {})[t] = account_fences(t, built.tiers[t], part, gated_by_tier[t])

    card_sha = short(sha256(inp.card))
    tiers_cfg = cfg["tiers"]
    manifest = {
        "story": "128.2",
        "pack_version": cfg["pack_version"],
        "inputs": {
            "card_sha": card_sha,
            "card": f"{cfg['sources']['card']['repo']}:{cfg['sources']['card']['path']}",
            "catalogue_sha": catalogue_sha,
            "card_snippet_sha256": sha256(inp.snippet),
            "repair_cases_sha256": sha256((PROMPTS / cfg["repair"]["cases"]).read_text(encoding="utf-8")),
            "errors_doc": "toke:docs/reference/errors.md (titles only)",
        },
        "tkc_pin": cfg["tkc_pin"],
        "tiers": {
            t: {"file": tiers_cfg[t]["file"], "title": tiers_cfg[t]["title"],
                "sha256": sha256(built.tiers[t]), "bytes": len(built.tiers[t].encode("utf-8")),
                "fences": gates["fences"][t], "card_gate": gates["card"][t]}
            for t in "abcd"
        },
        "exemplars": [{"id": e.id, "record": e.record, "source_sha256": e.source_sha256,
                       "patterns": e.patterns} for e in inp.exemplars],
        "bench128_exclude_families": sorted({re.sub(r"v\d+$", "", e.id) for e in inp.exemplars}),
        "repair_codes": gates["repairs"],
    }
    files = {tiers_cfg[t]["file"]: built.tiers[t] for t in "abcd"}
    files["MANIFEST.json"] = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    return files, manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-root", type=Path, default=WORKSPACE / "toke-corpus")
    ap.add_argument("--toke-root", type=Path, default=WORKSPACE / "toke")
    ap.add_argument("--tkc", type=Path, default=None,
                    help="the 131.103 pin (default $TOKE_TKC_PIN, else <workspace>/archive/pins/...)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild and gate, write nothing, exit 1 if a committed file differs")
    a = ap.parse_args(argv)
    cfg = json.loads(PACK_JSON.read_text(encoding="utf-8"))
    tkc = a.tkc or Path(os.environ.get("TOKE_TKC_PIN") or WORKSPACE / cfg["tkc_pin"]["path"])
    try:
        files, manifest = build(a.corpus_root, a.toke_root, tkc)
    except PackError as exc:
        print(f"build_prompt_pack: FAIL: {exc}", file=sys.stderr)
        return 1
    if a.check:
        drift = [n for n, text in files.items()
                 if not (PROMPTS / n).exists() or (PROMPTS / n).read_text(encoding="utf-8") != text]
        if drift:
            print(f"build_prompt_pack: STALE: {drift} differ from a fresh build; "
                  "run scripts/build_prompt_pack.py", file=sys.stderr)
            return 1
    else:
        for n, text in files.items():
            (PROMPTS / n).write_text(text, encoding="utf-8")
    t = manifest["tiers"]
    print(f"prompt pack {manifest['pack_version']}: card {manifest['inputs']['card_sha']} "
          f"catalogue {manifest['inputs']['catalogue_sha']} tkc {manifest['tkc_pin']['version']}")
    for k in "abcd":
        g = t[k]["card_gate"]
        print(f"  tier {k} {t[k]['file']}: {t[k]['bytes']} bytes, {t[k]['fences']} fences gated, "
              f"sha256 {short(t[k]['sha256'])}, card gate MISMATCH {g.get('MISMATCH')}")
    print(f"  exemplars {len(manifest['exemplars'])} built+run+lint-clean; "
          f"repair pairs {len(manifest['repair_codes'])} ({' '.join(manifest['repair_codes'])})")
    print("  " + ("up to date" if a.check else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
