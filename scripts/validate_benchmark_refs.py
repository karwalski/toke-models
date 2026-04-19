#!/usr/bin/env python3
"""Validate benchmark task reference programs: compile, execute, compare output.

Runs each reference_source from benchmark/tasks.jsonl through tkc, executes
the binary in a sandbox, and checks stdout against expected_output. Reports
which tasks have broken references so they can be excluded from eval.

Usage:
    python scripts/validate_benchmark_refs.py \
        --tasks benchmark/tasks.jsonl \
        --output benchmark/validation_report.json
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SANDBOX_CPU_SECONDS = 5
SANDBOX_MEM_BYTES = 256 * 1024 * 1024
SANDBOX_FSIZE_BYTES = 1 * 1024 * 1024
SANDBOX_NPROC = 0
SANDBOX_TIMEOUT_S = 10
COMPILE_TIMEOUT_S = 30


def _sandbox_preexec() -> None:
    resource.setrlimit(resource.RLIMIT_CPU,
                       (SANDBOX_CPU_SECONDS, SANDBOX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS,
                       (SANDBOX_MEM_BYTES, SANDBOX_MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (SANDBOX_FSIZE_BYTES, SANDBOX_FSIZE_BYTES))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (SANDBOX_NPROC, SANDBOX_NPROC))
    except (ValueError, OSError):
        pass


def compare_output(actual: str, expected: str) -> bool:
    actual_lines = actual.rstrip().split("\n")
    expected_lines = expected.rstrip().split("\n")
    if len(actual_lines) != len(expected_lines):
        return False
    for a, e in zip(actual_lines, expected_lines):
        a = a.rstrip()
        e = e.rstrip()
        if a == e:
            continue
        try:
            af, ef = float(a), float(e)
            if ef == 0.0:
                if abs(af) > 1e-6:
                    return False
            elif abs(af - ef) / abs(ef) > 1e-6:
                return False
        except ValueError:
            return False
    return True


def validate_one(task: dict) -> dict:
    source = task.get("reference_source", "")
    expected = task.get("expected_output", "")
    task_id = task.get("task_id", "?")
    stdin_input = task.get("stdin_input", "")

    with tempfile.TemporaryDirectory(prefix="tkref_") as tmpdir:
        src_path = os.path.join(tmpdir, "program.tk")
        bin_path = os.path.join(tmpdir, "program")

        with open(src_path, "w") as f:
            f.write(source)

        # Step 1: Compile
        try:
            comp = subprocess.run(
                ["tkc", src_path, "--out", bin_path],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
            )
        except FileNotFoundError:
            return {"task_id": task_id, "compiled": False, "error": "tkc not found"}
        except subprocess.TimeoutExpired:
            return {"task_id": task_id, "compiled": False, "error": "compile timeout"}

        if comp.returncode != 0 or not os.path.isfile(bin_path):
            return {
                "task_id": task_id,
                "compiled": False,
                "error": "compile failed",
                "diagnostics": comp.stderr.strip()[:500],
            }

        # Step 2: Execute in sandbox
        try:
            exe = subprocess.run(
                [bin_path],
                input=stdin_input,
                capture_output=True, text=True,
                timeout=SANDBOX_TIMEOUT_S,
                preexec_fn=_sandbox_preexec,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {
                "task_id": task_id,
                "compiled": True,
                "executed": False,
                "error": "execution timeout",
            }
        except OSError as e:
            return {
                "task_id": task_id,
                "compiled": True,
                "executed": False,
                "error": str(e),
            }

        stdout = exe.stdout
        exit_code = exe.returncode

        # Step 3: Compare
        has_expected = bool(expected.strip())
        if has_expected:
            output_match = compare_output(stdout, expected)
        else:
            output_match = exit_code == 0

        return {
            "task_id": task_id,
            "compiled": True,
            "executed": True,
            "exit_code": exit_code,
            "output_match": output_match,
            "actual_stdout": stdout.rstrip()[:500],
            "expected_stdout": expected.rstrip()[:500] if has_expected else "(none)",
            "stderr": exe.stderr.strip()[:200] if exe.stderr else "",
        }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON report (default: stdout summary only)")
    args = parser.parse_args()

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    print(f"Validating {len(tasks)} benchmark reference programs...")
    print(f"Sandbox: CPU={SANDBOX_CPU_SECONDS}s MEM={SANDBOX_MEM_BYTES//(1024*1024)}MB "
          f"timeout={SANDBOX_TIMEOUT_S}s")
    print()

    results = []
    n_compiled = 0
    n_executed = 0
    n_exit_zero = 0
    n_match = 0
    failures = []
    start = time.time()

    for i, task in enumerate(tasks):
        r = validate_one(task)
        results.append(r)

        if r.get("compiled"):
            n_compiled += 1
        if r.get("executed"):
            n_executed += 1
        if r.get("exit_code") == 0:
            n_exit_zero += 1
        if r.get("output_match"):
            n_match += 1
        else:
            failures.append(r)

        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            print(f"  [{i+1:4d}/{len(tasks)}] "
                  f"compile={n_compiled} exec={n_executed} "
                  f"exit0={n_exit_zero} match={n_match}")

    elapsed = time.time() - start
    total = len(tasks)

    print(f"\n{'='*60}")
    print(f"  Benchmark Reference Validation")
    print(f"{'='*60}")
    print(f"  Total tasks:    {total}")
    print(f"  Compiled:       {n_compiled}/{total} ({100*n_compiled/total:.1f}%)")
    print(f"  Executed:       {n_executed}/{total} ({100*n_executed/total:.1f}%)")
    print(f"  Exit zero:      {n_exit_zero}/{total} ({100*n_exit_zero/total:.1f}%)")
    print(f"  Output match:   {n_match}/{total} ({100*n_match/total:.1f}%)")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"{'='*60}")

    if failures:
        print(f"\n  Failed tasks ({len(failures)}):")

        # Categorize failures
        compile_fail = [f for f in failures if not f.get("compiled")]
        exec_fail = [f for f in failures if f.get("compiled") and not f.get("executed")]
        crash = [f for f in failures if f.get("executed") and f.get("exit_code", 0) != 0
                 and not f.get("output_match")]
        wrong_output = [f for f in failures if f.get("executed") and f.get("exit_code") == 0
                        and not f.get("output_match")]

        if compile_fail:
            print(f"\n  Compile failures ({len(compile_fail)}):")
            for f in compile_fail[:10]:
                diag = f.get("diagnostics", "")
                # Extract error code
                err = ""
                if "error_code" in diag:
                    import re
                    m = re.search(r'"error_code":"(E\d+)"', diag)
                    err = m.group(1) if m else ""
                elif "error[" in diag:
                    import re
                    m = re.search(r'error\[(E\d+)\]', diag)
                    err = m.group(1) if m else ""
                print(f"    {f['task_id']}: {err or diag[:80]}")
            if len(compile_fail) > 10:
                print(f"    ... and {len(compile_fail) - 10} more")

        if exec_fail:
            print(f"\n  Execution failures ({len(exec_fail)}):")
            for f in exec_fail[:10]:
                print(f"    {f['task_id']}: {f.get('error', '')}")

        if crash:
            print(f"\n  Runtime crashes ({len(crash)}):")
            for f in crash[:10]:
                print(f"    {f['task_id']}: exit={f.get('exit_code')} "
                      f"stderr={f.get('stderr', '')[:60]}")

        if wrong_output:
            print(f"\n  Wrong output ({len(wrong_output)}):")
            for f in wrong_output[:10]:
                actual = f.get("actual_stdout", "")[:60].replace("\n", "\\n")
                expected = f.get("expected_stdout", "")[:60].replace("\n", "\\n")
                print(f"    {f['task_id']}:")
                print(f"      expected: {expected}")
                print(f"      actual:   {actual}")

    if args.output:
        report = {
            "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": total,
            "compiled": n_compiled,
            "executed": n_executed,
            "exit_zero": n_exit_zero,
            "output_match": n_match,
            "elapsed_seconds": round(elapsed, 1),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report: {args.output}")

    return 0 if n_match == total else 1


if __name__ == "__main__":
    sys.exit(main())
