#!/usr/bin/env python3
"""Pass@1 evaluation harness for toke QLoRA adapters on CUDA.

Loads a PEFT QLoRA adapter, generates toke programs from benchmark task
descriptions, compiles with tkc, executes the binary in a sandbox, and
compares stdout to expected output. Reports Pass@1 (functional correctness)
and compile rate separately.

Designed for the EC2 A10G training instance. Run immediately after training
completes — the adapter is at output/7b-qlora-p2/adapter/.

Sandbox protections (Linux):
  - Isolated temp directory (auto-cleaned)
  - CPU time limit (5s via RLIMIT_CPU)
  - Virtual memory limit (256MB via RLIMIT_AS)
  - Max file write size (1MB via RLIMIT_FSIZE)
  - No child processes (RLIMIT_NPROC=0)
  - Network isolation via unshare --net (when available)
  - subprocess timeout as backstop

Usage:
    # Full evaluation (requires GPU + tkc in PATH):
    python scripts/eval_pass1_cuda.py \
        --adapter-dir output/7b-qlora-p2/adapter \
        --tasks benchmark/tasks.jsonl \
        --system-prompt corpus/system_prompt_phase2.txt

    # Dry-run (no model, synthetic results):
    python scripts/eval_pass1_cuda.py \
        --tasks benchmark/tasks.jsonl \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Sandbox configuration
# ---------------------------------------------------------------------------

SANDBOX_CPU_SECONDS = 5          # RLIMIT_CPU: hard wall on CPU time
SANDBOX_MEM_BYTES = 256 * 1024 * 1024   # RLIMIT_AS: 256 MB virtual memory
SANDBOX_FSIZE_BYTES = 1 * 1024 * 1024   # RLIMIT_FSIZE: 1 MB max file write
SANDBOX_NPROC = 0                # RLIMIT_NPROC: no fork/exec
SANDBOX_TIMEOUT_S = 10           # subprocess.run timeout (backstop)
COMPILE_TIMEOUT_S = 30           # tkc compilation timeout

# Whether unshare --net is available (Linux only, checked at startup).
_UNSHARE_NET: bool | None = None


def _check_unshare_net() -> bool:
    """Test whether unshare --net is available (requires Linux + CAP_SYS_ADMIN or user ns)."""
    global _UNSHARE_NET
    if _UNSHARE_NET is not None:
        return _UNSHARE_NET
    if platform.system() != "Linux":
        _UNSHARE_NET = False
        return False
    try:
        r = subprocess.run(
            ["unshare", "--net", "true"],
            capture_output=True, timeout=5,
        )
        _UNSHARE_NET = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _UNSHARE_NET = False
    return _UNSHARE_NET


def _sandbox_preexec() -> None:
    """Called in child process before exec. Sets resource limits."""
    # CPU time: hard kill after SANDBOX_CPU_SECONDS.
    resource.setrlimit(resource.RLIMIT_CPU,
                       (SANDBOX_CPU_SECONDS, SANDBOX_CPU_SECONDS))
    # Virtual memory.
    resource.setrlimit(resource.RLIMIT_AS,
                       (SANDBOX_MEM_BYTES, SANDBOX_MEM_BYTES))
    # Max file write size.
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (SANDBOX_FSIZE_BYTES, SANDBOX_FSIZE_BYTES))
    # No child processes (prevent fork bombs).
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (SANDBOX_NPROC, SANDBOX_NPROC))
    except (ValueError, OSError):
        pass  # Not all systems support setting NPROC to 0.


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> list[dict]:
    """Load benchmark tasks from JSONL.

    Expected fields per record:
      - task_id: str
      - description: str (the prompt for the model)
      - expected_output: str (stdout the compiled program should produce)
      - stdin_input: str (optional, fed to the program's stdin)
    """
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_system_prompt(path: Path) -> str:
    """Load the Phase 2 system prompt."""
    return path.read_text().strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(system: str, task: dict) -> str:
    """Build a ChatML prompt for the model."""
    desc = task.get("description", "")
    user_msg = (
        f"Write a complete toke program that solves the following task. "
        f"Output ONLY the toke source code, no explanation.\n\n{desc}"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def extract_toke_source(response: str) -> str:
    """Extract toke source from model response.

    Handles markdown fences, stop tokens, and explanation preamble.
    """
    text = response.strip()

    # Strip markdown fences if present.
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if lines and lines[0].strip().lower() in ("toke", ""):
                lines = lines[1:]
            candidate = "\n".join(lines).strip()
            if candidate.startswith("m="):
                return candidate
        if len(parts) >= 3:
            return parts[1].strip().split("\n", 1)[-1].strip()

    # Raw toke source — trim at stop tokens.
    if text.startswith("m="):
        for marker in ("<|im_end|>", "</s>", "<|endoftext|>"):
            if marker in text:
                text = text[:text.index(marker)]
        return text.strip()

    return text


# ---------------------------------------------------------------------------
# Compilation and sandboxed execution
# ---------------------------------------------------------------------------

def compile_toke(source_code: str, tmpdir: str) -> dict:
    """Compile toke source to a binary via tkc.

    Returns dict with keys: compiled, binary_path, diagnostics.
    """
    src_path = os.path.join(tmpdir, "program.tk")
    bin_path = os.path.join(tmpdir, "program")

    with open(src_path, "w") as f:
        f.write(source_code)

    try:
        result = subprocess.run(
            ["tkc", src_path, "--out", bin_path],
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
        )
        compiled = result.returncode == 0 and os.path.isfile(bin_path)
        return {
            "compiled": compiled,
            "binary_path": bin_path if compiled else None,
            "diagnostics": result.stderr.strip() if result.stderr else "",
        }
    except FileNotFoundError:
        return {
            "compiled": False,
            "binary_path": None,
            "diagnostics": "tkc not found in PATH",
        }
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "binary_path": None,
            "diagnostics": f"tkc timed out ({COMPILE_TIMEOUT_S}s)",
        }


def execute_sandboxed(binary_path: str, stdin_data: str = "",
                      timeout: int = SANDBOX_TIMEOUT_S) -> dict:
    """Execute a compiled toke binary inside a sandbox.

    Returns dict with keys: executed, exit_code, stdout, stderr, timed_out.
    """
    cmd: list[str] = []

    # Network isolation on Linux.
    if _check_unshare_net():
        cmd = ["unshare", "--net", binary_path]
    else:
        cmd = [binary_path]

    try:
        result = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_sandbox_preexec,
            cwd=os.path.dirname(binary_path),
        )
        return {
            "executed": True,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr.strip() if result.stderr else "",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "executed": True,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"execution timed out ({timeout}s)",
            "timed_out": True,
        }
    except OSError as e:
        return {
            "executed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "timed_out": False,
        }


def compare_output(actual: str, expected: str) -> bool:
    """Compare program output to expected, with whitespace normalization.

    Strips trailing whitespace from each line and trailing newlines.
    For floating-point tolerance, each line is compared: if both parse
    as floats, they pass if within 1e-6 relative tolerance.
    """
    actual_lines = actual.rstrip().split("\n")
    expected_lines = expected.rstrip().split("\n")

    if len(actual_lines) != len(expected_lines):
        return False

    for a, e in zip(actual_lines, expected_lines):
        a = a.rstrip()
        e = e.rstrip()
        if a == e:
            continue
        # Floating-point tolerance.
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


def evaluate_task(source_code: str, task: dict) -> dict:
    """Full evaluation pipeline for one task: compile, execute, compare.

    Returns a dict with compile/execute/pass status and diagnostics.
    """
    expected_output = task.get("expected_output", "")
    stdin_input = task.get("stdin_input", "")
    has_expected = bool(expected_output.strip())

    with tempfile.TemporaryDirectory(prefix="toke_eval_") as tmpdir:
        # Step 1: Compile.
        comp = compile_toke(source_code, tmpdir)
        if not comp["compiled"]:
            return {
                "compiled": False,
                "executed": False,
                "exit_zero": False,
                "output_match": False,
                "passed": False,
                "diagnostics": comp["diagnostics"],
                "actual_stdout": "",
            }

        # Step 2: Execute in sandbox.
        exe = execute_sandboxed(comp["binary_path"], stdin_data=stdin_input)
        if not exe["executed"]:
            return {
                "compiled": True,
                "executed": False,
                "exit_zero": False,
                "output_match": False,
                "passed": False,
                "diagnostics": exe["stderr"],
                "actual_stdout": "",
            }

        exit_zero = exe["exit_code"] == 0

        # Step 3: Compare output.
        if has_expected:
            output_match = compare_output(exe["stdout"], expected_output)
        else:
            # No expected output: pass if compiled and exited cleanly.
            output_match = exit_zero

        passed = comp["compiled"] and exit_zero and output_match

        return {
            "compiled": True,
            "executed": True,
            "exit_zero": exit_zero,
            "output_match": output_match,
            "passed": passed,
            "diagnostics": exe["stderr"] if exe["stderr"] else comp["diagnostics"],
            "actual_stdout": exe["stdout"].rstrip(),
        }


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(adapter_dir: Path | None, model_base: str):
    """Load quantised base model, optionally with PEFT QLoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"  Base model: {model_base}")
    print(f"  Adapter: {adapter_dir or '(none — base model only)'}")

    tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single completion (greedy)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", type=Path, default=None,
                        help="Path to trained PEFT adapter directory")
    parser.add_argument("--model-base", type=str,
                        default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--tasks", required=True, type=Path,
                        help="Benchmark tasks JSONL file")
    parser.add_argument("--system-prompt", type=Path, default=None,
                        help="System prompt text file (defaults to built-in)")
    parser.add_argument("--output-dir", type=Path, default=Path("eval-results"),
                        help="Output directory for predictions and summary")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (for quick spot-checks)")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only check compilation, skip execution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic predictions, no model loading")
    args = parser.parse_args(argv)

    # Load tasks.
    if not args.tasks.exists():
        print(f"ERROR: tasks file not found: {args.tasks}", file=sys.stderr)
        return 1
    tasks = load_tasks(args.tasks)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print(f"Loaded {len(tasks)} benchmark tasks")

    # Check sandbox capabilities.
    if not args.compile_only and not args.dry_run:
        if _check_unshare_net():
            print("Sandbox: unshare --net available (network isolated)")
        else:
            print("Sandbox: unshare --net unavailable (no network isolation)")
        print(f"Sandbox: CPU={SANDBOX_CPU_SECONDS}s, MEM={SANDBOX_MEM_BYTES // (1024*1024)}MB, "
              f"FSIZE={SANDBOX_FSIZE_BYTES // 1024}KB, NPROC={SANDBOX_NPROC}, "
              f"timeout={SANDBOX_TIMEOUT_S}s")

    # Verify tkc is available.
    if not args.dry_run:
        tkc_path = shutil.which("tkc")
        if tkc_path:
            print(f"tkc: {tkc_path}")
        else:
            print("ERROR: tkc not found in PATH", file=sys.stderr)
            return 1

    # Load system prompt.
    system_prompt = ""
    if args.system_prompt and args.system_prompt.exists():
        system_prompt = load_system_prompt(args.system_prompt)
        print(f"System prompt: {len(system_prompt)} chars")
    else:
        print("WARNING: no system prompt provided", file=sys.stderr)

    # Load model.
    model, tokenizer = None, None
    if not args.dry_run:
        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer(args.adapter_dir, args.model_base)
        print("Model loaded.\n")

    # Run evaluation.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    n_passed = 0
    n_compiled = 0
    n_executed = 0
    n_exit_zero = 0
    n_output_match = 0
    start = time.time()

    with open(predictions_path, "w") as out_f:
        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{i}")

            if args.dry_run:
                import random
                rng = random.Random(hash(task_id) + 42)
                passed = rng.random() < 0.65
                compiled = passed or rng.random() < 0.20
                result = {
                    "task_id": task_id,
                    "prediction": f"m=eval;f=main():$i64{{<0}};",
                    "compiled": compiled,
                    "executed": compiled,
                    "exit_zero": passed,
                    "output_match": passed,
                    "passed": passed,
                    "diagnostics": "",
                    "actual_stdout": "",
                    "dry_run": True,
                }
            else:
                prompt = build_prompt(system_prompt, task)
                raw_response = generate(model, tokenizer, prompt,
                                        max_new_tokens=args.max_tokens)
                source = extract_toke_source(raw_response)

                if args.compile_only:
                    with tempfile.TemporaryDirectory(prefix="toke_eval_") as tmpdir:
                        comp = compile_toke(source, tmpdir)
                    eval_result = {
                        "compiled": comp["compiled"],
                        "executed": False,
                        "exit_zero": False,
                        "output_match": False,
                        "passed": comp["compiled"],
                        "diagnostics": comp["diagnostics"],
                        "actual_stdout": "",
                    }
                else:
                    eval_result = evaluate_task(source, task)

                result = {
                    "task_id": task_id,
                    "prediction": source,
                    "raw_response": raw_response,
                    **eval_result,
                }

            out_f.write(json.dumps(result) + "\n")
            if result.get("passed"):
                n_passed += 1
            if result.get("compiled"):
                n_compiled += 1
            if result.get("executed"):
                n_executed += 1
            if result.get("exit_zero"):
                n_exit_zero += 1
            if result.get("output_match"):
                n_output_match += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                elapsed_so_far = time.time() - start
                rate = (i + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:4d}/{len(tasks)}] "
                      f"pass={n_passed} compile={n_compiled} "
                      f"exec_ok={n_exit_zero} "
                      f"({rate:.1f} tasks/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    total = len(tasks)
    pass_at_1 = n_passed / total if total else 0.0
    compile_rate = n_compiled / total if total else 0.0
    exec_rate = n_exit_zero / total if total else 0.0
    match_rate = n_output_match / total if total else 0.0

    # Print results.
    print(f"\n{'='*60}")
    print(f"  Pass@1 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Tasks:          {total}")
    print(f"  Compile rate:   {compile_rate:.4f} ({n_compiled}/{total})")
    print(f"  Exit-zero rate: {exec_rate:.4f} ({n_exit_zero}/{total})")
    print(f"  Output match:   {match_rate:.4f} ({n_output_match}/{total})")
    print(f"  Pass@1:         {pass_at_1:.4f} ({n_passed}/{total})")
    print(f"  Elapsed:        {elapsed:.1f}s ({elapsed/60:.1f}m)")
    if not args.dry_run:
        print(f"  Adapter:        {args.adapter_dir}")
    print(f"  Predictions:    {predictions_path}")
    print(f"{'='*60}")

    # Write summary.
    summary = {
        "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_base": args.model_base,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "tasks_file": str(args.tasks),
        "n_tasks": total,
        "pass_at_1": round(pass_at_1, 4),
        "compile_rate": round(compile_rate, 4),
        "exit_zero_rate": round(exec_rate, 4),
        "output_match_rate": round(match_rate, 4),
        "n_passed": n_passed,
        "n_compiled": n_compiled,
        "n_exit_zero": n_exit_zero,
        "n_output_match": n_output_match,
        "elapsed_seconds": round(elapsed, 1),
        "max_new_tokens": args.max_tokens,
        "temperature": 0.0,
        "compile_only": args.compile_only,
        "dry_run": args.dry_run,
        "sandbox": {
            "cpu_seconds": SANDBOX_CPU_SECONDS,
            "mem_mb": SANDBOX_MEM_BYTES // (1024 * 1024),
            "fsize_kb": SANDBOX_FSIZE_BYTES // 1024,
            "nproc": SANDBOX_NPROC,
            "timeout_s": SANDBOX_TIMEOUT_S,
            "unshare_net": _check_unshare_net() if not args.dry_run else None,
        },
        "gate": "Gate 2",
        "gate_thresholds": {
            "pass_at_1_minimum": 0.75,
            "compilation_rate_minimum": 0.95,
        },
        "gate_result": "PASS" if (pass_at_1 >= 0.75 and compile_rate >= 0.95) else "FAIL",
    }
    summary_path = args.output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:        {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
