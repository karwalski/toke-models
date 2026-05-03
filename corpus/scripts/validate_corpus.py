#!/usr/bin/env python3
"""validate_corpus.py — Run tkc --check on every program in a corpus and collect pass/fail statistics.

Accepts a directory of .tk files or a JSONL file with a "code" field per line.
Writes validated.jsonl, rejected.jsonl, and validation_report.json to the output directory.

Usage:
    python validate_corpus.py --input corpus/default/ --output corpus/validated/ --tkc ./tkc --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ProgramEntry:
    """A single program to validate."""
    source: str          # origin file path or JSONL path:line
    code: str
    extra: dict = field(default_factory=dict)  # other JSONL fields preserved


@dataclass
class ValidationResult:
    """Result of running tkc --check on one program."""
    source: str
    code: str
    passed: bool
    exit_code: int | None = None
    errors: list[dict] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    timed_out: bool = False
    crashed: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def iter_programs(input_path: Path) -> Iterator[ProgramEntry]:
    """Yield ProgramEntry objects from a directory of .tk files or a JSONL file."""
    if input_path.is_dir():
        tk_files = sorted(input_path.rglob("*.tk"))
        for fp in tk_files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"warning: cannot read {fp}: {exc}", file=sys.stderr)
                continue
            yield ProgramEntry(source=str(fp), code=text)
    elif input_path.is_file() and input_path.suffix in (".jsonl", ".json"):
        with open(input_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warning: {input_path}:{lineno}: bad JSON: {exc}", file=sys.stderr)
                    continue
                if "code" not in obj:
                    print(f"warning: {input_path}:{lineno}: missing 'code' field", file=sys.stderr)
                    continue
                code = obj.pop("code")
                yield ProgramEntry(
                    source=f"{input_path}:{lineno}",
                    code=code,
                    extra=obj,
                )
    else:
        sys.exit(f"error: --input must be a directory of .tk files or a .jsonl file, got: {input_path}")


def count_programs(input_path: Path) -> int:
    """Count programs without loading code (for --dry-run)."""
    if input_path.is_dir():
        return sum(1 for _ in input_path.rglob("*.tk"))
    elif input_path.is_file() and input_path.suffix in (".jsonl", ".json"):
        count = 0
        with open(input_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "code" in obj:
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count
    return 0


# ---------------------------------------------------------------------------
# Validation worker
# ---------------------------------------------------------------------------

_TKC_PATH: str = ""
_TIMEOUT: int = 5


def _init_worker(tkc_path: str, timeout: int) -> None:
    """Initialiser for each pool worker — sets globals."""
    global _TKC_PATH, _TIMEOUT
    _TKC_PATH = tkc_path
    _TIMEOUT = timeout


def _validate_one(entry_tuple: tuple[str, str, dict]) -> ValidationResult:
    """Validate a single program in its own temp file. Runs inside a worker process."""
    source, code, extra = entry_tuple

    tmp_dir = tempfile.mkdtemp(prefix="tkc_val_")
    tmp_file = os.path.join(tmp_dir, "prog.tk")
    try:
        with open(tmp_file, "w", encoding="utf-8") as fh:
            fh.write(code)

        try:
            proc = subprocess.run(
                [_TKC_PATH, "--check", "--diag-json", tmp_file],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                source=source,
                code=code,
                passed=False,
                timed_out=True,
                error_codes=["TIMEOUT"],
                extra=extra,
            )
        except OSError as exc:
            return ValidationResult(
                source=source,
                code=code,
                passed=False,
                crashed=True,
                error_codes=["CRASH"],
                errors=[{"message": str(exc)}],
                extra=extra,
            )

        passed = proc.returncode == 0
        errors: list[dict] = []
        error_codes: list[str] = []

        if not passed:
            # Parse JSON diagnostics from stdout (one JSON object per line).
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    diag = json.loads(line)
                    errors.append(diag)
                    if "code" in diag:
                        error_codes.append(str(diag["code"]))
                except json.JSONDecodeError:
                    pass

            # If no structured diagnostics, try stderr as a single JSON array.
            if not errors and proc.stderr.strip():
                try:
                    diag_list = json.loads(proc.stderr.strip())
                    if isinstance(diag_list, list):
                        for diag in diag_list:
                            errors.append(diag)
                            if "code" in diag:
                                error_codes.append(str(diag["code"]))
                    elif isinstance(diag_list, dict):
                        errors.append(diag_list)
                        if "code" in diag_list:
                            error_codes.append(str(diag_list["code"]))
                except json.JSONDecodeError:
                    errors.append({"raw_stderr": proc.stderr.strip()})

            # Detect crashes (signal-based exit codes).
            if proc.returncode and proc.returncode < 0:
                return ValidationResult(
                    source=source,
                    code=code,
                    passed=False,
                    exit_code=proc.returncode,
                    errors=errors,
                    error_codes=error_codes if error_codes else ["CRASH"],
                    crashed=True,
                    extra=extra,
                )

        return ValidationResult(
            source=source,
            code=code,
            passed=passed,
            exit_code=proc.returncode,
            errors=errors,
            error_codes=error_codes,
            extra=extra,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Error-category helpers
# ---------------------------------------------------------------------------

_ERROR_CATEGORIES = {
    "E1": "lexer",
    "E2": "parser",
    "E3": "names",
    "E4": "types",
    "E5": "semantic",
}


def categorise_error(code: str) -> str:
    """Map an error code like E1042 to its category."""
    for prefix, category in _ERROR_CATEGORIES.items():
        if code.startswith(prefix):
            return category
    return "other"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(results: list[ValidationResult], elapsed: float) -> dict:
    """Build the validation_report.json summary."""
    total = len(results)
    pass_count = sum(1 for r in results if r.passed)
    fail_count = total - pass_count
    timeout_count = sum(1 for r in results if r.timed_out)
    crash_count = sum(1 for r in results if r.crashed)

    all_error_codes: list[str] = []
    category_counts: Counter[str] = Counter()
    for r in results:
        if not r.passed:
            all_error_codes.extend(r.error_codes)
            for ec in r.error_codes:
                category_counts[categorise_error(ec)] += 1

    code_counter = Counter(all_error_codes)

    return {
        "total_programs": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_count / total, 4) if total else 0.0,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "fail_by_category": dict(category_counts.most_common()),
        "top_10_error_codes": [
            {"code": code, "count": cnt}
            for code, cnt in code_counter.most_common(10)
        ],
        "all_error_codes": dict(code_counter.most_common()),
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(results: list[ValidationResult], output_dir: Path, report: dict) -> None:
    """Write validated.jsonl, rejected.jsonl, and validation_report.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    validated_path = output_dir / "validated.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "validation_report.json"

    with open(validated_path, "w", encoding="utf-8") as vf, \
         open(rejected_path, "w", encoding="utf-8") as rf:
        for r in results:
            entry: dict = {"source": r.source, "code": r.code}
            entry.update(r.extra)
            if r.passed:
                vf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                entry["errors"] = r.errors
                entry["error_codes"] = r.error_codes
                if r.timed_out:
                    entry["timed_out"] = True
                if r.crashed:
                    entry["crashed"] = True
                rf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  validated.jsonl : {report['pass_count']} programs")
    print(f"  rejected.jsonl  : {report['fail_count']} programs")
    print(f"  validation_report.json written")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run tkc --check on every program in a corpus and collect pass/fail statistics.",
    )
    p.add_argument(
        "--input", required=True, type=Path,
        help="Directory of .tk files or a JSONL file with 'code' field per line.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for validated.jsonl, rejected.jsonl, validation_report.json.",
    )
    p.add_argument(
        "--tkc", default="tkc", type=str,
        help="Path to the tkc compiler binary (default: 'tkc' on PATH).",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count).",
    )
    p.add_argument(
        "--timeout", type=int, default=5,
        help="Timeout in seconds per program (default: 5).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Count programs without running tkc.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve tkc path.
    tkc_path = args.tkc
    if not shutil.which(tkc_path):
        resolved = shutil.which(tkc_path)
        if resolved is None:
            # Try as an explicit file path.
            if not os.path.isfile(tkc_path):
                print(f"error: tkc compiler not found at '{tkc_path}'", file=sys.stderr)
                print("  Provide --tkc /path/to/tkc or ensure tkc is on your PATH.", file=sys.stderr)
                return 1
    else:
        tkc_path = shutil.which(tkc_path) or tkc_path

    # Validate input exists.
    if not args.input.exists():
        print(f"error: input path does not exist: {args.input}", file=sys.stderr)
        return 1

    # Dry run: count only.
    if args.dry_run:
        n = count_programs(args.input)
        print(f"dry-run: {n} programs found in {args.input}")
        return 0

    # Load programs.
    programs = list(iter_programs(args.input))
    if not programs:
        print(f"warning: no programs found in {args.input}", file=sys.stderr)
        return 0

    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    num_workers = min(num_workers, len(programs))

    print(f"Validating {len(programs)} programs with {num_workers} workers (timeout={args.timeout}s)...")

    # Prepare entries as tuples for pickling.
    entries = [(p.source, p.code, p.extra) for p in programs]

    start = time.monotonic()

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tkc_path, args.timeout),
    ) as pool:
        results: list[ValidationResult] = pool.map(_validate_one, entries)

    elapsed = time.monotonic() - start

    report = build_report(results, elapsed)

    # Print summary.
    print(f"\nValidation complete in {elapsed:.1f}s")
    print(f"  Total : {report['total_programs']}")
    print(f"  Pass  : {report['pass_count']} ({report['pass_rate'] * 100:.1f}%)")
    print(f"  Fail  : {report['fail_count']}")
    if report["timeout_count"]:
        print(f"  Timeout: {report['timeout_count']}")
    if report["crash_count"]:
        print(f"  Crash  : {report['crash_count']}")
    if report["top_10_error_codes"]:
        print("\n  Top error codes:")
        for item in report["top_10_error_codes"]:
            cat = categorise_error(item["code"])
            print(f"    {item['code']} ({cat}): {item['count']}")

    print(f"\nWriting output to {args.output}/")
    write_outputs(results, args.output, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
