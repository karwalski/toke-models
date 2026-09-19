# Epic 128 — Pre-registered evaluation protocol (training reset)

**Story:** 128.1 · **Status:** PRE-REGISTERED · **Date:** 2026-09-19
**Compiler basis:** tkc 2.8.0 · **Card basis:** syntax card v2 (catalogue `6dad9e907cfb`, 46 entries)
**Gates:** 128.2–128.9, lanes 128.10–128.14, scorecard 128.15, export 131.29
**Output consumers:** `toke-model/docs/lane-scorecard.md` (128.15), the 128.9 decision ADR

---

## 0. The lock rule

This document is **append-only from the moment the first Lane A number is published**
(§7.1, gate G0). Section 5 (benchmark), Section 6 (metrics and the definition of a pass),
Section 7 (the Lane A floor and its margin) and Section 9 (fixed vs variable) cannot be
changed after that point. Clarifications go in the amendment log (§13) with a date and a
reason; **thresholds and margins cannot be amended at all**.

The precedent is `toke/docs/spec/gate3-criteria.md`, which was pre-registered on 2026-05-24
under the same rule and honoured it: its single amendment (2026-05-25, the `io.readln`
correction) explicitly left every threshold untouched. That is the standard here.

The reason for the rule is specific and is worth stating plainly: Epic 128 runs six lanes
that compete for the same conclusion, one of which (Lane A) is the no-training baseline. If
the comparison is not fixed in advance, a disappointing lane can be rescued by moving its
goalposts — a different task subset, a different temperature, a kinder pass definition, a
best-of-N sample relabelled Pass@1. Every one of those moves is available today in the
existing harnesses (K10), and this project has already shipped two results that a
pre-registration would have caught: a best-of-N score reported under `pass_at_1` (K10), and
a Gate 2 declared PASS against the loosest of three differently-numbered thresholds that
were live at the time (K11). Pre-registration is the only control that survives an owner who
wants a lane to win.

---

## 1. What this protocol decides, and what it does not

**Decides.** The benchmark set and its provenance; how disjointness from the Epic 131
tokenizer holdout is verified mechanically; the metrics and the exact computation of each,
including what counts as a pass; the falsifiable condition every trained lane must clear
against Lane A, and the pre-declared meaning of no lane clearing it; the stage gates at
which every lane is measured; and what is held fixed so the lanes are comparable.

**Does not decide.** Which base model wins (128.3), whether to fine-tune or train from
scratch (128.5/128.6), which tokenizer is right (128.12/116.9), what the pipeline shape
should be (128.7), or whether to train at all. Those are outcomes of running this protocol,
and 128.9 reads them off the scorecard rather than arguing them.

**Does not authorise.** Any training run or any compute provisioning. 128.4 owns that, and
nothing in this document is a licence to start.

---

## 2. Inputs consolidated

| Input | Path | What it contributes |
|---|---|---|
| Epic 128 rows 128.1–128.15 | `toke/docs/progress.md` | Lane definitions, the scorecard metric list, the "no lane advances without beating Lane A" rule |
| Tokenizer v0.4 plan | `toke/docs/architecture/tokenizer-v04-plan.md` | D5 corpus rule, D6 `--min` canonical form, D7 family-level holdout and baseline-first gating |
| TEMSpec v1.1 | `toke-spec/docs/temspec.md` | Metric definitions, tokenizer pinning, bootstrap CI rules, §6.2 lane-crossing prohibition |
| Landscape review | `toke/docs/about/reviews/landscape-2026-09-18.md` (= `~/tk/input/research and recommendations 20260918.md`, plus an editor's note) | Lane A's rationale, RLVR as the gap Gate 2 exposed, cost-per-solved-task as the unit, the tokenizer-free hedge |
| Repositioning brief | `toke/docs/about/positioning-2026-09.md` §5, §8 | The durable claim; falsification tests F1–F6, of which **F3 is the Lane A rule this protocol operationalises** |
| Canonical facts | `toke/docs/about/canonical.md`, `canonical.json` | The only sanctioned counts and gate numbers |
| Metrics baseline | `toke/docs/metrics-baseline.md` | The authoritative measured position; what is withdrawn |
| Prior pre-registration | `toke/docs/spec/gate3-criteria.md` | Evaluation-protocol shape inherited wholesale (§6); its absolute thresholds are **not** inherited (K3) |
| Prior training plan | `toke/docs/spec/training-next-phase.md` | The v0.3-phase LOCKED table (reopened, §3), the ≤3-round repair cap, quality≫quantity, GRPO/RLVR |
| From-scratch architecture | `toke/docs/spec/training-architecture-v2.md` | Corpus-volume estimates for a from-scratch lane; the seven mandatory per-program gates |
| Gate 2 record | `archive/toke-legacy-20260819/docs/spec/gate2-decision.md` (local archive) | The actual Gate 2 numbers. **Not at the path Epic 128 cites** — see conflict K8 |
| Reasoning channel | `toke/docs/spec/reasoning-channel.md` | `.tkc.md` out-of-band channel; what is and is not counted |
| Tokenizer holdout | `toke-corpus/scripts/curate_tokenizer_set.py`, `data/tokenizer_manifest_pre131.json` | The 131.22 holdout, its family rule, and its (vacuous) benchmark-disjointness assertion |
| Corpus freeze | `toke-corpus/regen/AUDIT_129.md` (reopened), successor `AUDIT_131.md` (131.27, pending) | The training pool and its provenance |

---

## 3. Carries forward vs reopened

Epic 128's preamble states that nothing previously decided is binding: the LOCKED table in
`training-next-phase.md` was locked for the v0.3 phase only. This section is the explicit
ledger 128.1 owes, so 128.9 can record which reopened decisions get re-locked.

### 3.1 Reopened — no prior status carries any weight

| v0.3-phase decision | Prior status | Reopened by |
|---|---|---|
| v0.3 syntax frozen | LOCKED | v0.4 shipped; moot |
| Purpose-built BPE, 16K vocab | LOCKED | 128.12 (superword vs bespoke vs base vs base+added), 116.9; 131.20 measured the shipped 8k at **1.154× cl100k** on v0.4 text (N = 2,000) |
| No inline comments / reasoning-light corpus | LOCKED | 131.53 A/B; the review names it a conflict with efficient-reasoning evidence. **The language does not change** — only the record shape |
| Qwen 2.5 Coder 7B base | OPEN | 128.3 bake-off |
| GRPO/RLVR vs SFT-only | OPEN | 128.11 (Lane B) is now a first-class lane |
| Strict vs adaptive curriculum | OPEN | 128.5, 128.14 |
| Whether to train a model at all | never asked | **128.10 + 128.15** — this is the new question, and §7 is its decision rule |
| Gate-3 absolute thresholds (≥35% / >50% / ≥85% functional) | one PRE-REGISTERED, two proposed | K3: not inherited. Epic 128 scores relative to Lane A, not against a 2026-05 absolute |

### 3.2 Carries forward unchanged

- **The ≤3-round repair cap.** Four concordant sources, no conflicting number anywhere:
  `training-next-phase.md` rec #4 (+6pp / +1.7pp / +1pp per round, arXiv 2511.03898), its
  weeks-4–6 plan, and progress rows 128.7 and 128.15.
- **Binary scoring.** Compile AND all test cases pass = 1; anything else = 0
  (`gate3-criteria.md` §2). No partial credit anywhere in this protocol.
- **Execution over plausibility.** No LLM judge, no rubric score, no `judge_score` field may
  contribute to any pass/fail metric. They may be reported as diagnostics only. Grounded in
  131.42: 429 of 576 A-ERR corpus records passed their tests by returning the harness marker
  as a string — passing a test is not evidence of correctness when the test can be gamed.
- **The `.tkc.md` reasoning channel** as the out-of-band mechanism; only `.tk` source is
  counted in any efficiency number.
- **TEMSpec §6.2**: one tokenizer on both sides of any cross-language comparison, always.
- **Quality ≫ quantity** for SFT data selection.
- **The CI-gated project facts** (59 charset / 14 keywords / 56 stdlib modules / 228
  conformance cases), enforced by `toke/scripts/verify_project_facts.py`.

---

## 4. Conflicts between the planning inputs

Named, not silently resolved. Where this protocol picks a side it says so and why; where it
cannot, it says that instead.

### K1 — Gate 2 functional correctness: 8% vs 55.6% vs 2.2%

The landscape review (and therefore the argument in rows 128.10 and 128.11) reasons from
"about 8% functional correctness". `gate2-decision.md`, `metrics-baseline.md`,
`canonical.md` and `gate3-criteria.md` all record **55.6% (272/489)**, corrected on
2026-05-25 after an `io.readln` glue defect was fixed — "the 8% figure reflected
infrastructure failure, not model failure". `positioning-2026-09.md` states the correction
explicitly and notes the review's *conclusion* survives it while its *force* does not.

A third number exists and is lower than both: the full-local re-audit of all 1,748 v0.3.9
corpus programs gives **37.5% compile (655/1,748) and ~2.2% fully correct (38 PASS)**. The
55.6% is on a curated 500-hidden + 200-eval set the model was optimised against.

**Resolution.** This protocol quotes Gate 2 as *100% compile / 55.6% functional on a
curated set, with an un-curated floor of 37.5% / 2.2%* — never one without the other, per
`metrics-baseline.md` caveat 2. The review's 8% is not used anywhere. The denominator
ambiguity is unresolved and stays that way: N = 489 is never reconciled to the 500 hidden or
the 700 total in any document, so no rate derived from it is reproducible. **No Gate 2
number is a target in this protocol** — it is history, and §7 sets the bar relative to a
Lane A measured now.

### K2 — Sampling temperature for the headline Pass@1

`toke-spec/docs/gate-criteria.md` (Gate 2 evaluation protocol): **greedy, T = 0**.
`toke/docs/spec/gate3-criteria.md` (the locked pre-registration): **T = 0.2, n = 1**.

**Resolution.** The headline is **T = 0, n = 1, greedy**, because it is deterministic and
therefore reproducible from the gate card, which T = 0.2 is not. **T = 0.2, n = 1 is also
run and reported** as a secondary row, so continuity with the Gate 3 pre-registration is
preserved and neither number can be chosen after the fact — both are always published.

### K3 — Which functional threshold Epic 128 inherits

Four numbers for the same gate across three documents: **>50%** (`training-next-phase.md`
"Gate 3 Criteria (Proposed)"), **≥35%** (`gate3-criteria.md` C1, PRE-REGISTERED),
**≥85%** primary / ≥60% on additional families (`gate-criteria.md` Gate 3), and **≥50% on
2+ families** (`training-next-phase.md` weeks 13–28).

**Resolution.** None is inherited. Every one was set against a different set, a different
syntax version and a different harness, and the Gate-3 programme they belong to never ran
(`metrics-baseline.md`: "2026-08 — no model gate ran"). Epic 128's bar is **relative to Lane
A measured on BENCH-128 under tkc 2.8.0** (§7). An absolute threshold set in May 2026 cannot
be honest about a benchmark defined in September 2026.

### K4 — The tokenizer: LOCKED 16K vs the 8,192 actually trained vs +15.4%

`training-next-phase.md` locks "Purpose-built BPE — 16K vocab" and names the tokenizer among
the project's strongest parts. Gate 2 actually trained **8,192**. `metrics-baseline.md`
measures the shipped SentencePiece 8k at **279,672 tokens vs cl100k's 242,427 on the same
2,000 `--min`+masked records — 15.4% *more*** — and the only 16,384 artefact
(`tokenizer_v03`) is lossy (drops 2,606 `\` via a null `unk_token`), so its apparent 0.545
ratio is an artefact, not a win.

**Resolution.** The "LOCKED 16K" row is void (§3.1). 128.12 decides the tokenizer on
downstream correctness after an identical fine-tune, not on compression alone — SuperBPE's
headline was compression *with* an accuracy gain, so a compression-only comparison decides
nothing.

### K5 — "Token efficiency vs baseline" as a 128.1 metric

Row 128.1 lists "token efficiency vs baseline" among the metrics to pre-register. Read as
toke-vs-Python it collides with TEMSpec §6.2 and with the measured position: under one
shared tokenizer toke costs **1.34× [1.22, 1.48]** the cl100k tokens of equivalent Python
(N = 60).

**Resolution.** In this protocol "token efficiency vs baseline" means **same-language,
cross-lane**: tokens per solved task for lane L against tokens per solved task for Lane A,
on the same tasks, in the same tokenizer lane. It never means toke vs Python. Cross-language
density remains informational and remains governed by TEMSpec §2.3 and §6.2.

### K6 — The tokenizer holdout's benchmark-disjointness assertion is vacuous

`curate_tokenizer_set.py` asserts `hold_ids & bench_ids == ∅` and
`hold_bases & bench_ids == ∅` against `toke-corpus/data/holdout_task_ids.txt`, then records
`disjoint_benchmark: true` in the manifest. That file holds **974 bare integers and 164
`HumanEval/N` ids** — an external-benchmark namespace. The v0.4 corpus id namespace is
`A-CAT-NNNNvMM` / `D-CAT-NNNNvMM` / library ids. Measured: the intersection of that file
with all 23,382 `MANIFEST.jsonl` task_ids is **0**, and with the 3,040 derived bases is
**0**. The assertion cannot fail, so the manifest flag records that a check ran, not that
anything was verified.

Additionally `read_id_file()` returns an empty set for a missing path, logging a warning —
so a typo'd `--benchmark-ids` silently produces an unprotected carve that still stamps
`disjoint_benchmark: true`.

**Consequence for this protocol.** Disjointness is verified by **content hash and family id
across namespaces**, plus an explicit **vacuity guard** (§5.3 check D6). Filed as a story
(§14, S5).

### K7 — The existing "200 eval" set is corpus-derived and cannot be held out

Row 128.1 says the Gate-2 500-hidden/200-eval set "needs a v0.4 refresh pass". The 500
hidden maps to `toke-eval/benchmark/hidden_tests/task-a-0501…1000.yaml`. The 200 eval most
plausibly maps to `toke-model/benchmark/tasks.jsonl` — and that file is generated *from*
`toke-corpus/data/corpus_default.jsonl`, with `task_id` = `"A-" + <corpus task_id> + "-" +
<8 hex>`. Measured at family level after stripping the affixes:

- `tasks.jsonl` (200 tasks, 161 bases): **160 of 161 bases** intersect the 3,040 v0.4 corpus
  families; **7** intersect the 661 families of the `pre131` tokenizer holdout.
- `tasks_v2.jsonl` (400 tasks, 272 bases): **272 of 272** intersect the corpus families;
  **19** intersect the tokenizer holdout families.

**Resolution.** These files are **not** eligible for BENCH-128 and are quarantined (§14, S4).
A refresh pass cannot fix them: they are derived from the training pool by construction. The
"v0.4 refresh" obligation in row 128.1 is discharged by building BENCH-128 (§5) instead.

### K8 — A cited input does not exist at its cited path

Epic 128's preamble cites `spec/gate2-decision.md`. No such file exists in `toke/docs/spec/`.
The only copy is at `archive/toke-legacy-20260819/docs/spec/gate2-decision.md` (local archive). Recorded
here so the citation resolves; the archive copy is the one this protocol read.

### K9 — `tokenizer_manifest_v04.json` does not exist yet

Row 128.1's hard constraint names `tokenizer_manifest_v04.json` (131.22). On disk there is
only the `pre131` label — `data/tokenizer_manifest_pre131.json` and
`data/tokenizer_holdout_pre131.txt` (1,131 records / 661 families). The `v04` run is gated on
131.19.

**Resolution.** Disjointness is verified against **`pre131` now and re-verified against `v04`
when 131.22 completes**, and the re-verification is a **blocking gate on the first training
run of any lane**, not on this document (§5.3, gate G0). Both manifest SHAs are recorded in
every scorecard row.

### K10 — Harness defects that would corrupt a comparison

- `toke-eval/toke_eval/pass_at_k.py::run_tests()` reads `tests["test_cases"]`; the hidden-test
  YAMLs use `test_inputs`. It returns `(0, 0)` when the key is absent — it would score every
  task 0/0 and report it as a result.
- `toke-eval/benchmark/run_benchmark.py --n-samples N` takes the **best of N** samples
  (`break` on the first perfect sample) and reports it under `pass_at_1`. That is pass@N, not
  Pass@1.
- `toke-eval/benchmark/harness/{run,score,report}.py` are **0-byte stubs**.
- `toke-eval/benchmark/tasks/` contains only `schema.json` — no task files; the harness must
  be pointed at `hidden_tests/`.

`toke-eval/scripts/pass_at_k.py` is correct (Chen et al. unbiased estimator) and is the only
Pass@k implementation this protocol permits. The rest are filed (§14, S3).

### K11 — Gate 2's pass criterion: three thresholds, passed against the loosest

`toke-spec/docs/gate-criteria.md` set Gate 2 at **Pass@1 ≥ 75%** on held-out tasks, plus ten
further criteria (token reduction ≥ 15%, corpus ≥ 100K, ≥ 3 tokenizers, ≥ 200 aligned tasks,
Gate 1.5 complete, …). The Gate 2 decision document's own results table states the target as
**> 70%**. The criterion actually used to declare PASS was a third one, from Epic 10.10:
*"7B fine-tuned model outperforms baseline on toke code generation using default syntax"* —
no numeric floor at all. None of the other ten `gate-criteria.md` criteria is addressed in
the decision document.

**Resolution.** Nothing is inherited, and the episode is the reason §0 exists. Epic 128 has
**one** advancement condition (§7.1), stated numerically, fixed before any result, with a
pre-declared meaning for its failure (§7.3). A lane cannot be passed against whichever of
several live thresholds it happens to clear, because there is only one.

---

## 5. The benchmark set — BENCH-128

### 5.1 Composition and provenance

BENCH-128 is **750 tasks in two strata**, frozen as one artefact.

**S1 — 500 procedural tasks, `task-a-0501` … `task-a-1000`.**
Source: `toke-eval/benchmark/hidden_tests/`, written by
`toke-eval/benchmark/generate_tasks.py` (`START_ID = 501`, `COUNT = 500`, `random.seed(42)`
at import, `random.Random(2024)` in `main()`), from a fixed template list with per-template
generator functions. Each task carries `{id, phase, category, description, input_type,
output_type, test_inputs:[{input, expected}]}`, `test_inputs` with `minItems: 20` per
`tasks/schema.json`.

Why this stratum: it was generated procedurally from templates, never from the toke corpus,
so it is **structurally independent of the training pool** rather than merely filtered
against it. Its id namespace (`task-a-NNNN`) cannot collide with corpus ids, which is a
weakness for id-based checks (K6) and exactly why §5.3 checks content.

Excluded from S1: `task-a-0001` … `task-a-0060`. Those 60 were re-delivered on v0.4 and
their repaired solutions are **published**; `toke/docs/about/toke-eval-drift-decision.md`
states they may not be used as a held-out evaluation set for any model that has seen them.
They stay out permanently.

**S2 — 250 v0.4-native task families carved from the frozen corpus.**
Source: the freeze-131 pool (`AUDIT_131.md`, 131.27; fallback the freeze-129 snapshot
preserved by 131.12). Eligibility, all conditions required:

1. `audit_bucket == "pass"` under AUDIT_131 — the record executes and its tests pass;
2. its a_tests were authored or re-verified under 131.47/131.49 (the 129-era gates checked
   printed output against test cases whose own arity and expectations were wrong);
3. **A-ERR records only from the 147 that 131.42 found correct** — the 429 that fake their
   error case by returning the harness marker as a `str` are excluded outright;
4. its family is **not** in the 131.22 tokenizer holdout, **not** in the tokenizer train
   split's protected set, and **not** among the 2,000 ids of the 131.20 baseline sample
   (`toke-tokenizer/data/baseline_sample_ids_v04.txt`);
5. the record survives `tkc --min` under a pinned binary and is pattern-lint clean.

Carve: **whole families**, stratified by `category × difficulty`, using the existing
`carve_holdout()` algorithm from `toke-corpus/scripts/curate_tokenizer_set.py` with
`seed = 128` (a different seed from 131.22's `seed = 131`, so the two carves cannot
coincide), `target = "families"`, exactly 250 families. Family rule is the project's
existing one:

```python
_BASE = re.compile(r"^([A-Z]-[A-Z]+-\d+)v\d+$")   # D-WEB-0006v87 -> D-WEB-0006
```

Ids that do not match are their own family. Whole families only: variants of one base task
share structure, so splitting a family across benchmark and training leaks.

Why this stratum is necessary: S1's generator restricts I/O to integer, list and boolean
types — there is **no string I/O anywhere in the 500 procedural tasks**. A model scored only
on S1 would be scored on a fraction of what toke is for. S2 supplies string handling,
parsing, error propagation, CLI and file I/O, in v0.4 syntax, with execution-verified tests.

**S2 is removed from every lane's training data.** The 131.29 export
(`toke-model/training-data-v04/`) is gated on this protocol precisely so this exclusion is
applied at the point of export rather than trusted afterwards. Removing 250 of the ~4,384
families in the curated pool costs the training set roughly 6% of its families; that cost is
accepted.

### 5.2 Freeze

BENCH-128 is frozen as `toke-eval/benchmark/bench128/` containing:

- `manifest.json` — per task: `task_id`, `stratum` (`S1` | `S2`), `category`, `difficulty`,
  `base` (family), `n_cases`, `prompt_sha256`, `tests_sha256`, and for S2 the source
  `record_sha256` and `min_sha256` carried over from the corpus manifest;
- `prompts.jsonl` — the exact prompt text presented to every lane;
- header fields: `protocol_sha256` (of this file), `tkc_bin_sha` + `tkc_version` from
  `tkc_pin.py`, `card_sha` and `catalogue_sha` of syntax card v2, the corpus freeze tag, the
  tokenizer manifest SHAs (`pre131` and, once it exists, `v04`), and `bench_sha256` over the
  sorted task list.

`bench_sha256` goes in every gate card and every scorecard row. A run whose `bench_sha256`
differs from the frozen value is not comparable and is not admitted to the scorecard.

Test *content* stays where it is. `hidden_tests/` is write-only for humans (AGENTS.md §3.5):
agents never read it, the scoring harness does, and the disjointness checker reads it only
to emit hashes and counts.

### 5.3 How disjointness is verified — mechanically, not asserted

A single checker, `toke-eval/scripts/check_bench_disjoint.py`, exits non-zero on any
violation and writes `bench128_disjoint.json`. It runs at gate G0, again whenever any input
artefact changes, and in CI.

Inputs: `bench128/manifest.json`; `toke-corpus/data/tokenizer_manifest_<label>.json`
(`records[]` carry `task_id`, `base`, `split`, `record_sha256`, `min_sha256`,
`line_sha256`); `data/tokenizer_holdout_<label>.txt` and `data/tokenizer_training_<label>.txt`
(one masked `--min` program per line, **no ids** — which is why the content check exists);
`toke-tokenizer/data/baseline_sample_ids_v04.txt`; the corpus `MANIFEST.jsonl`; and
`toke-eval/benchmark/solutions/*.toke`.

| # | Check | Computation | Failure |
|---|---|---|---|
| D1 | Id disjointness | `bench_ids ∩ tokholdout_ids = ∅`; `bench_ids ∩ baseline_ids = ∅` | hard |
| D2 | **Family** disjointness | `{base_of(t) : t ∈ bench} ∩ {r.base : r.split == "holdout"} = ∅`, same `_BASE` regex as the corpus | hard |
| D3 | Family integrity | no BENCH-128 family appears in any lane's training export; `bench_families ∩ export_families = ∅` | hard |
| D4 | **Content disjointness across namespaces** | for every BENCH-128 task, canonicalise its reference/solution text with the pinned `tkc --min` and the same escape-aware `"_"` string mask the tokenizer curation uses, take `sha256`; intersect with the line hashes of *both* `tokenizer_holdout_*.txt` and `tokenizer_training_*.txt`, and with `sha256` of `--min`-canonicalised `benchmark/solutions/*.toke` | hard |
| D5 | Near-duplicate | MinHash (128 permutations) over 5-gram shingles of the masked `--min` text, **Jaccard ≥ 0.80** against every holdout and training line; plus TF-IDF cosine ≥ 0.85 over task *descriptions* via `toke-corpus/registry/firewall.py::check_similarity()` | hard — the pair is reported and the BENCH-128 member is removed and replaced from the same stratum before the freeze; after the freeze it is a hard stop |
| D6 | **Vacuity guard** | for each of D1/D2, assert both operand sets are non-empty **and** that their id shapes are drawn from a common namespace (same regex class). A check whose operands cannot intersect by construction is reported as `VACUOUS`, never `PASS` | hard |
| D7 | Input-presence guard | every named id/manifest file must exist and parse; a missing file is an error, never an empty set | hard |
| D8 | Determinism | re-running the checker on unchanged inputs reproduces `bench128_disjoint.json` byte-for-byte | hard |

D4 is the check that does the real work. D1 and D2 compare ids, and BENCH-128's S1 stratum
lives in a namespace (`task-a-NNNN`) that can never collide with corpus ids — which is
exactly the failure mode K6 documents in the existing tooling. D6 makes that failure mode
impossible to mistake for a pass: a check that could not have failed is reported as vacuous.

`bench128_disjoint.json` records, for every check, the two set sizes, the intersection size,
the namespace classes compared, and the verdict. "Verified" means that file exists, is
current for the input SHAs, and every verdict is `PASS`.

### 5.4 Statistical adequacy

N = 750 binary outcomes. The 95% bootstrap half-width on a proportion near 0.5 is ≈ 3.6pp
unpaired and tighter paired, so the 5pp margin in §7.1 is detectable at this N. All CIs:
**10,000 bootstrap resamples, percentile method, paired by task id** (TEMSpec §5.2). Every
headline number is reported with its CI and its N, per stratum and pooled. A lane that wins
pooled but loses on a stratum has that stated in its scorecard row.

---

## 6. Metrics

Each metric below is computed by one implementation, under one pinned compiler, in one
sandbox. The scoring code's git SHA is recorded in every run.

### 6.0 Generation conditions (identical for every lane)

Headline: **T = 0, greedy, n = 1, one attempt** — strict Pass@1, no best-of-N.
Secondary: **T = 0.2, n = 1** (K2). Diagnostic only: **n = 10 at T = 0.8** for Pass@k (§6.3).
`max_new_tokens = 1024`; a truncated generation is a failure, not a retry.
Prompt: the frozen `prompts.jsonl` text plus the declared system-prompt tier (§9).

### 6.1 M1 — compile Pass@1

Per task: extract the toke source from the response; write it; run the **pinned** `tkc
--check`; then `tkc --out <bin> <src>`. `compile = 1` iff `--check` exits 0 **and** the
binary exists. Compile timeout 30s. Aggregate = mean over 750 tasks, with CI.

A lint failure is **not** a compile failure; lint violations per solution are reported
separately (§6.8).

### 6.2 M2 — functional Pass@1 (the primary correctness metric)

Per task, for every test case: run the compiled binary in the sandbox with the case's input;
`pass_case = 1` iff **exit code 0** and stdout, after normalisation, equals the expected
output exactly. Normalisation: strip trailing whitespace per line; exactly one trailing
newline; no other transformation; **no float tolerance** unless the task declares one in its
record, in which case the declared tolerance is part of `tests_sha256`.

`functional = 1` iff `compile == 1` **and every** case passes. Binary, all-or-nothing, per
`gate3-criteria.md` §2. Aggregate = mean over 750 tasks, with CI.

Sandbox, identical for every lane (the limits already implemented in
`toke-model/scripts/eval_pass1_cuda.py`): `RLIMIT_CPU` 5s, `RLIMIT_AS` 256 MB, `RLIMIT_FSIZE`
1 MB, `RLIMIT_NPROC` 0, network isolation where available, 10s wall-clock backstop, isolated
auto-cleaned temp dir. A timeout, a crash, a non-zero exit and a wrong answer are all
failures and are all recorded distinctly in the per-task CSV.

**Pass@1 is on execution, never on plausibility.** No judge score, no similarity to a
reference, no "looks right". And no test may be passable without solving the task: S2
inherits the 131.42 exclusion, and any BENCH-128 task later found gameable is removed by
amendment with every affected result re-scored and republished (§13).

### 6.3 M3 — Pass@k, diagnostic only

k ∈ {1, 5, 10} from n = 10 samples at T = 0.8, using the unbiased estimator
`pass@k = 1 − C(n−c, k)/C(n, k)` (Chen et al. 2021) as implemented in
`toke-eval/scripts/pass_at_k.py` — the only permitted implementation (K10). Pass@k never
substitutes for M2 and is never reported as Pass@1.

### 6.4 M4 — tokens and bytes per solved task

```
tokens_per_solved(lane, L) = Σ tokens over ALL attempts, including failed ones
                             and every repair round
                             ÷ number of tasks solved (M2 == 1)
```

Reported in **every** lane of 131.50 — `proxy8k`, `byte256` (raw bytes), byte-level patch
count, `v03`, `qwen25coder`, `cl100k`, `o200k` — with the tokenizer artefact SHA and package
version pinned per TEMSpec §3.2. **A token number without its lane does not ship.**
`bytes_per_solved` is computed identically on `byte256` and is the unit-agnostic sibling
(§10). Denominators are solved tasks, not attempts: a lane that emits fewer tokens by failing
more often does not win this metric.

### 6.5 M5 — cost and energy per solved task

Cost-of-Pass accounting (arXiv 2504.13359) per 131.55: price every attempt including failed
ones and every repair round; price input cache-aware (cached reads ≈ 0.1× uncached, per the
dated price table stored beside the scorecard); for trained lanes, amortise training compute
over a declared task volume stated in the row, and also report un-amortised inference cost
so the two are separable.

Energy is **estimated, with the method stated in the row**: GPU-hours × device TDP × PUE for
local and trained lanes; a vendor median-prompt figure for API lanes. Never presented as
measured.

### 6.6 M6 — repair-loop convergence

Rounds 0–3, cap 3. Report `solved@0, solved@1, solved@2, solved@3` and the marginal gain per
round. A lane that has not converged by round 3 is scored at round 3.

**What the loop may see:** the compiler's `--diag-json` diagnostics (including the `fix`
field, which is populated only where deterministic); the program's own stdout and stderr;
and *which* cases failed together with their inputs. **What it may never see:** the expected
output of any test case. A loop shown the expected output is copying an answer, and any run
that does so is void.

### 6.7 M7 — first-shot validity under constrained decoding

Fraction of raw samples that parse with zero diagnostics, measured **with and without** the
131.52 grammar mask, plus mask-construction cost and per-token overhead. Every lane reports
both, so the constrained-decoding factor never silently favours Lane A.

### 6.8 M8 — terseness with structure (the property toke actually claims)

The claim under test is terseness *plus enough structure for reliable generation*, so both
halves are measured, on **correct solutions only** (measuring compactness over wrong answers
rewards nonsense):

- `min_bytes` of correct solutions — median and p99;
- pattern-lint violations per correct solution, gross and net of exemptions (131.10);
- fraction of correct solutions in the catalogue's canonical form;
- a **reliability–terseness pair** reported together as `(M2, median min_bytes)`. A lane that
  wins bytes and loses M2 has not won. If 131.54 finds terseness costs correctness, that
  finding lands here as a reported tension, not as a quiet reweighting.

---

## 7. Lane A is the floor

Lane A (128.10) is a **frontier model + syntax card v2 + the 131.52 grammar artefacts, with
no training of any kind**. It exists because the landscape review's central challenge is that
constrained decoding plus an existing model reproduces most of the compile-guarantee benefit
with no training at all, and because we have our own evidence it works in context (agent
workers on the syntax card bank ~98% of a shard; Anka reports 99.9% parse success with zero
prior exposure). This protocol treats that challenge as the null hypothesis.

### 7.1 The falsifiable condition

For a trained lane L at a given stage gate, on BENCH-128, against Lane A measured at the
same gate with the same `bench_sha256`:

> **L clears the floor iff BOTH hold:**
>
> **(a) Correctness.** `M2(L) − M2(A) ≥ 0.05` (5 percentage points) **and** the 95%
> bootstrap CI of the paired per-task difference `M2(L) − M2(A)` (10,000 resamples,
> percentile, paired by `task_id`) **excludes 0**.
>
> **(b) Economics.** `M5(L) ≤ M5(A)` on cost per solved task, and the 95% bootstrap CI of the
> ratio `M5(L)/M5(A)` **excludes 1 from above** — i.e. L being more expensive is ruled out at
> 95%, not merely unproven.
>
> A lane that clears (a) but not (b) is recorded **CORRECT-BUT-COSTLY**; a lane that clears
> (b) but not (a) is recorded **CHEAP-BUT-NOT-BETTER**. Neither advances. Both are published.

The 5pp margin is fixed here, before any result is seen, and cannot be amended (§0). It is
not a number derived from a prior gate — those are all void (K3) — it is the smallest
difference this N can resolve with the CI rule above, chosen so that "beats Lane A" cannot be
satisfied by noise.

### 7.2 Anti-rescue clauses

1. Lane A is **re-measured at every stage gate** with the same card v2 `card_sha`, the same
   grammar artefact SHAs and the same prompt tier. Its number is published before any trained
   lane is scored at that gate.
2. Lane A may **not be weakened** after any lane's result is seen: no downgrade of its model,
   no reduction of its prompt tier, no disabling of constrained decoding, no reduction of its
   repair budget. Lane A gets the same ≤3 repair rounds every other lane gets.
3. If Lane A's frontier model is upgraded or its prompt pack improves, **every lane
   re-benchmarks against the upgraded Lane A** before any advancement decision. A trained lane
   never keeps a win earned against a weaker baseline.
4. No lane may substitute a subset of BENCH-128, a different temperature, a different pass
   definition or a best-of-N sample for the headline comparison.
5. Every touch of BENCH-128 is logged (§8.3) and every result is published, including results
   the lane's owner dislikes.

### 7.3 If no trained lane clears the floor — pre-declared

This outcome has a meaning fixed now, so it cannot be explained away later.

> **NO-TRAIN.** If, at the final stage gate, no lane among 128.5, 128.6, 128.11, 128.12,
> 128.13 and 128.14 clears §7.1 against Lane A, then the bespoke-model programme **stops**.
> 128.9 is written as a NO-TRAIN ADR recording that training was not justified on the
> evidence. The deliverable becomes the grammar artefacts, the compiler and its diagnostics,
> the syntax card and pattern catalogue, and the MCP tooling — shipped and documented as the
> product. The tokenizer programme (116.9) descopes to whatever the 128.12 lane needs and no
> further. The result is **published as a negative result**, per the falsifiability principle
> already in `gate-criteria.md` and already written into row 128.15 and F3 of
> `positioning-2026-09.md` §8.

NO-TRAIN is not a failure of Epic 128. It is one of the two answers the epic exists to
produce, and it is the cheaper one. A protocol that could only conclude "train something"
would not be worth pre-registering.

Three secondary outcomes are also pre-declared:

- **PARTIAL.** Exactly one lane clears §7.1 and it is a cheap lane (128.11 RLVR, or 128.14
  continued-pretrain/adapters). 128.9 re-locks only that lane and its stated scope; the
  expensive lanes (128.6 from-scratch, bespoke tokenizer) stay closed.
- **AMBIGUOUS.** A lane clears (a) with a CI whose lower bound is below 5pp only because of
  one stratum. The row records it, and the lane runs one — and only one — additional
  replication at the final gate with a pre-declared seed, published whatever it shows.
- **LANE A WINS OUTRIGHT.** Lane A beats every trained lane on both axes. Same consequence as
  NO-TRAIN, stated more strongly in the ADR.

---

## 8. Stage gates

Every lane is benchmarked at every stage. "Measured only at the end" is how a lane's owner
discovers too late that its first checkpoint was already behind.

### 8.1 The gates

| Gate | When | What is measured | Rule |
|---|---|---|---|
| **G0 Readiness** | Before any lane starts | BENCH-128 frozen; `check_bench_disjoint.py` all-PASS against `pre131` **and** `v04` once it exists; harness defects K10 fixed; **Lane A measured and published** | No lane starts before G0 closes. The `v04` re-verification blocks the first training run of any lane |
| **G1 Inputs** | After data/prompt/tokenizer prep, before training compute | Full scorecard row from the lane's *starting* model, untrained, with the lane's own prompt and tokenizer | Detects a lane that has already lost on its inputs |
| **G2 Early checkpoint** | ~25% of the lane's planned steps | Full scorecard row | **Early-stop:** if the 95% CI *upper* bound of `M2(L)` is below the point estimate of `M2(A)`, the lane is STOPPED and recorded STOPPED, not deleted |
| **G3 Mid** | ~50% | Full scorecard row | Trend recorded; a lane whose M2 falls between G2 and G3 must state why in the row |
| **G4 Final** | End of training | Full scorecard row; §7.1 evaluated | The advancement decision |
| **G5 Loop** | G4 model + ≤3-round repair loop | M6, plus M2 at each round | Reported separately; a lane may not fold repair gains into its G4 number |
| **G6 Constrained** | G4 model + grammar mask | M7 with and without | Reported separately |

Lane A is measured at G0 and re-measured at G4 (and at any gate where its model or prompt
changed, per §7.2.3).

### 8.2 The scorecard row

One row per (lane, gate), appended to `toke-model/docs/lane-scorecard.md`, never edited:

`lane · gate · date · protocol_sha · bench_sha · tkc_bin_sha · card_sha · catalogue_sha ·
tokenizer_manifest_sha · model id + weights sha · decoding params · M1 [CI] · M2 [CI] ·
M2 by stratum · M3 (k=1,5,10) · M4 per lane (7 columns) · bytes/solved · M5 cost [CI] ·
energy (method) · M6 solved@0..3 · M7 with/without · M8 (min_bytes p50/p99, lint/solution) ·
floor verdict (CLEARS / CORRECT-BUT-COSTLY / CHEAP-BUT-NOT-BETTER / BELOW / STOPPED) ·
per-task CSV path`

A run without a complete row is not a result. Per TEMSpec §5.3, the raw per-task CSV is
published for every run so any aggregation can be independently recomputed. Each run also
files a gate card on `toke-eval/gate_card_template.md`.

### 8.3 Benchmark access ledger

BENCH-128 is touched **at most once per lane per gate**. Every touch appends to
`bench128_access.jsonl`: lane, gate, date, `bench_sha`, model id, and the resulting M1/M2.
Every touch is published, including ones whose results are bad. Iteration happens on
**BENCH-128-DEV**, a separate development slice drawn from families that are in the training
pool and in neither BENCH-128 nor the tokenizer holdout; DEV never appears in a scorecard
row and never in the 128.9 ADR.

---

## 9. What is fixed and what a lane may vary

### 9.1 Fixed — identical for every lane; changing any of these invalidates the comparison

Benchmark set and `bench_sha256` · the prompt text in `prompts.jsonl` · the system-prompt
tier used for scoring (**tier (b), full syntax card v2 including the Patterns block**, the
128.2 pack sha-stamped; other tiers may be explored in DEV and reported as diagnostics but
never in a headline row) · the pinned `tkc` binary (`tkc_pin.py`, `tkc_bin_sha` stamped in
every row; the pin exists because the `tkc` symlink is relinked by any concurrent `make`) ·
stdlib version · sandbox limits · scoring code and its git SHA · headline decoding
parameters (T = 0, n = 1, greedy, `max_new_tokens = 1024`) · the repair-loop shape (≤3
rounds; diagnostics and failing-case inputs visible, expected outputs never) · the token lane
set and every tokenizer artefact SHA · the dated price table and energy factors · the
bootstrap procedure (10,000 resamples, percentile, paired) · the pass definition in §6.2 ·
the 5pp margin in §7.1.

### 9.2 Variable — the independent variable each lane is testing

Base model and size (128.3, 128.14) · tokenizer (128.12: bespoke BPE / superword / base
model's own / base + added tokens) · training objective and method (SFT, QLoRA/DoRA, GRPO/
RLVR, continued pretraining, from scratch) · training data mix and curriculum, **drawn only
from the freeze-131 TRAIN split with BENCH-128 families excluded** · adapter rank and
hyperparameters · generation architecture (token-level, byte-level, dynamic chunking —
128.13) · pipeline shape (single model vs frontier-reasons/toke-writes — 128.7).

### 9.3 Declared factors — varied, but reported both ways by every lane

Constrained decoding on/off (§6.7) · the `.tkc.md` reasoning channel present/absent (131.53)
· T = 0 and T = 0.2 (K2). A lane may not report only its favourable setting.

### 9.4 Never variable

The benchmark, the definition of a pass, the margin, the thresholds, the CI procedure, and
the meaning of NO-TRAIN.

---

## 10. Durability: if token-level generation stops being how code models work

The protocol is written so that a shift to byte-level or dynamic-chunking models
(H-Net-class, BLT-class) costs it **columns, not thresholds**.

1. **Every decision rule in §7 is stated on unit-free quantities.** M2 is a count of solved
   tasks. M5 is money and energy per solved task. Neither mentions a token. If the BPE token
   stops being the unit of account, §7.1 is still computable, unchanged, and still decides.
2. **Token counts are already plural.** M4 reports every 131.50 lane including raw bytes
   (`byte256`) and byte-level patch counts, so the pivot is a change of which column is
   quoted, not a re-measurement.
3. **Pre-declared unit-migration rule.** If the serving architecture a lane targets does not
   consume BPE tokens, its BPE columns become informational and **bytes and byte-level
   patches per solved task** take their place in the row. No threshold changes, because no
   threshold is expressed in tokens. This rule is declared now so that the migration cannot
   look like a lane rescuing itself with a friendlier unit.
4. **The structural properties are measured directly.** M7 (first-shot validity and mask cost
   under a grammar constraint) and M8 (canonical-form adherence, lint violations) are
   properties of the grammar and the canonical form, not of any tokenizer. They are what
   `positioning-2026-09.md` §5 identifies as surviving a tokenizer-free world, and 131.51 is
   the spike that tests whether they actually do.
5. **The claim under test is measured as stated.** toke's case is terseness *plus* enough
   structure for reliable generation. M8 reports compactness and reliability as a pair, on
   correct solutions only, so a lane cannot buy one with the other.

If 131.51 retires the "durable under tokenizer-free architectures" claim (F1 in
`positioning-2026-09.md` §8), this protocol does not change: M4's byte lanes become
informational, and §7 decides exactly as before.

---

## 11. Relationship to Epic 131 and the corpus freeze

- **Training pool:** the 131.27 re-freeze (`AUDIT_131.md`), fallback the freeze-129 snapshot
  preserved by 131.12. AUDIT_131 supersedes AUDIT_129; the 129 freeze's claim that "every
  record that CAN be executed passes" did not hold (131.42, 131.47, 131.40, 131.44).
- **Tokenizer holdout:** 131.22, `pre131` today, `v04` pending 131.19. BENCH-128 is disjoint
  from it by §5.3 and by a different carve seed.
- **Export:** 131.29 applies the BENCH-128 exclusion list at export time. That story is
  already gated on 128.1 for exactly this reason.
- **Card and catalogue:** 128.2's prompt pack derives from syntax card v2 + the pattern
  catalogue (131.11), not the 129-era card. The card and catalogue SHAs are fixed inputs.
- **Grammar artefacts:** 131.52 supplies Lane A's constrained-decoding artefacts and M7's
  mask-cost numbers. Lane A cannot be measured before 131.52 lands, which is why it sits at
  gate G0.
- **Cost harness:** 131.55 supplies M5. Its four-arm comparison (toke + our model, toke +
  frontier + constrained decoding, Python + frontier, Python + frontier + type-constrained
  decoding) is the wider context; BENCH-128 supplies arms one and two.

---

## 12. What would falsify this protocol itself

- A BENCH-128 task is found to be passable without solving it (a 131.42-class gamed test).
  Consequence: the task is removed by amendment, **every affected result is re-scored and
  republished**, and the amendment log states how many rows moved.
- `check_bench_disjoint.py` reports `VACUOUS` on D1 or D2 after the freeze. Consequence: all
  results since the last all-PASS run are quarantined until the check is repaired.
- Lane A cannot be reproduced within its own CI on a re-run at the same gate with the same
  SHAs. Consequence: the floor is unmeasurable that day; no advancement decision is taken.
- The pass definition is found to admit a solution that hardcodes the test inputs.
  Consequence: an argv/hardcoding detector is added as a **reported diagnostic** and the
  affected tasks are amended. It is not retrofitted as a threshold — the prior 67%
  hardcoding baseline was withdrawn (132.14) and has no replacement measurement, so no
  threshold may be set against it.

---

## 13. Amendment log

Append-only. Thresholds and margins cannot be amended.

| Date | Amendment | Reason |
|---|---|---|
| 2026-09-19 | Pre-registered. | Initial. |

---

## 14. Stories this protocol requests

Filed as requests to the main thread; agents never edit `toke/docs/progress.md`.

| Ref | Story | Priority |
|---|---|---|
| S1 | **128.1a — Build and freeze BENCH-128.** Carve S2 (250 families, seed 128, `target="families"`, eligibility per §5.1), assemble `toke-eval/benchmark/bench128/{manifest.json,prompts.jsonl}`, stamp `bench_sha256` and every input SHA. Blocks every lane. | **P0** |
| S2 | **128.1b — `check_bench_disjoint.py` with the vacuity guard.** Checks D1–D8 per §5.3, `bench128_disjoint.json`, non-zero exit, wired into CI. Blocks G0. | **P0** |
| S3 | **128.1c — Repair the eval harness before it is trusted.** `toke_eval/pass_at_k.py` reads `test_cases` where the hidden tests use `test_inputs` and silently scores 0/0; `run_benchmark.py --n-samples` reports best-of-N under `pass_at_1`; `benchmark/harness/{run,score,report}.py` are 0-byte stubs; `benchmark/tasks/` holds only `schema.json`. | **P0** |
| S4 | **128.16 — Quarantine the contaminated model benchmark sets.** `toke-model/benchmark/tasks.jsonl` (160/161 bases intersect the v0.4 corpus families, 7 intersect the tokenizer holdout) and `tasks_v2.jsonl` (272/272 and 19) are corpus-derived and cannot be held out for any lane trained on the corpus. Banner them, stop `eval_pass1_cuda.py` defaulting to them, point it at BENCH-128. | **P1** |
| S5 | **131.22a — The tokenizer holdout's benchmark-disjointness assertion is vacuous.** `curate_tokenizer_set.py` asserts against `holdout_task_ids.txt` (974 integers + 164 `HumanEval/N`), which shares no namespace with corpus ids — intersection with all 23,382 task_ids and all 3,040 bases is 0, so `disjoint_benchmark: true` records nothing. Also: `read_id_file()` silently returns an empty set for a missing path while the flag still stamps `true`. Add a content-hash check and a vacuity guard; re-stamp the manifest. | **P1** |
| S6 | **128.17 — Publish the Lane A card at G0.** Lane A measured, gate card filed, scorecard row appended, before any other lane starts. | **P0** |
| S7 | **128.1d — BENCH-128-DEV slice + `bench128_access.jsonl` ledger.** The iteration set and the access log that keeps the benchmark from being consumed by tuning. | **P1** |
| S8 | **128.18 — Fix the Epic 128 preamble citation.** `spec/gate2-decision.md` exists only under the local `archive/toke-legacy-20260819/` tree; the row should cite the archive path or mark the input archived. | **P2** |

---

## 15. Pre-registration statement

The benchmark set, the metrics, the definition of a pass, the Lane A floor condition, its
5pp margin, the stage gates and the meaning of NO-TRAIN are fixed as of **2026-09-19**,
before any Epic 128 lane has produced a result and before any training compute has been
provisioned. No number in this document was chosen with knowledge of any lane's outcome.
