#!/usr/bin/env python3
"""Evaluate a trained LoRA/DoRA adapter on benchmark tasks.

Loads a trained adapter, generates predictions on benchmark tasks,
computes Pass@1 via tkc --check, and outputs predictions JSONL
for downstream comparison.

Usage:
    python scripts/eval_adapter.py \
        --adapter-dir adapters/dora/ \
        --benchmark-dir eval/benchmark/ \
        --output predictions.jsonl \
        --model-base Qwen/Qwen2.5-Coder-7B-Instruct

    # Dry-run mode (no model loading, synthetic results):
    python scripts/eval_adapter.py \
        --adapter-dir adapters/dora/ \
        --benchmark-dir eval/benchmark/ \
        --output predictions.jsonl \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def discover_tasks(benchmark_dir: Path) -> list[dict]:
    """Discover benchmark tasks from YAML or JSONL files.

    Each task must have at minimum a task_id and a prompt.
    Supports both YAML task files (one per file) and a single JSONL file.
    """
    tasks = []

    # Check for tasks.jsonl
    jsonl_file = benchmark_dir / "tasks.jsonl"
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    # Check for individual YAML task files
    try:
        import yaml
    except ImportError:
        yaml = None

    if yaml:
        for yaml_file in sorted(benchmark_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                task = yaml.safe_load(f)
            if task and "task_id" in task:
                tasks.append(task)

    # Check for individual JSON task files
    for json_file in sorted(benchmark_dir.glob("*.json")):
        if json_file.name == "metadata.json":
            continue
        with open(json_file) as f:
            task = json.load(f)
        if task and "task_id" in task:
            tasks.append(task)

    return tasks


def generate_prediction_mlx(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    """Generate a prediction using the loaded MLX model."""
    from mlx_lm import generate as mlx_generate

    response = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.0,  # greedy for evaluation
    )
    return response


def check_with_tkc(source_code: str) -> dict:
    """Run tkc --check on generated source code.

    Returns dict with 'compiled' (bool), 'passed' (bool), and 'diagnostics' (str).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toke", delete=False) as f:
        f.write(source_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["tkc", "--check", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        compiled = result.returncode == 0
        return {
            "compiled": compiled,
            "passed": compiled,
            "diagnostics": result.stderr.strip() if result.stderr else "",
        }
    except FileNotFoundError:
        return {
            "compiled": False,
            "passed": False,
            "diagnostics": "tkc not found in PATH",
        }
    except subprocess.TimeoutExpired:
        return {
            "compiled": False,
            "passed": False,
            "diagnostics": "tkc --check timed out (30s)",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def count_tokens(text: str) -> int:
    """Approximate token count (whitespace-split as fallback)."""
    return len(text.split())


def generate_dry_run_prediction(task: dict, seed: int) -> dict:
    """Generate synthetic prediction for dry-run mode.

    Uses deterministic randomness based on task_id hash for reproducibility.
    Simulates a ~60% pass rate with varying token counts.
    """
    rng = random.Random(hash(task.get("task_id", "")) + seed)
    passed = rng.random() < 0.60
    compiled = passed or rng.random() < 0.20  # some compile but fail tests
    token_count = rng.randint(30, 200)

    return {
        "task_id": task.get("task_id", "unknown"),
        "prompt": task.get("prompt", ""),
        "prediction": f"// dry-run synthetic prediction for {task.get('task_id', '')}",
        "compiled": compiled,
        "passed": passed,
        "token_count": token_count,
        "diagnostics": "" if compiled else "dry-run: synthetic compile failure",
        "dry_run": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adapter-dir", required=True, type=Path,
                        help="Path to trained adapter directory")
    parser.add_argument("--benchmark-dir", required=True, type=Path,
                        help="Path to benchmark tasks directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output predictions JSONL path")
    parser.add_argument("--model-base", type=str,
                        default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model name (default: Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per generation (default: 512)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate predictions with synthetic pass rates (no model loading)")
    args = parser.parse_args(argv)

    # Discover tasks
    if not args.benchmark_dir.exists():
        print(f"ERROR: benchmark directory not found: {args.benchmark_dir}", file=sys.stderr)
        return 1

    tasks = discover_tasks(args.benchmark_dir)
    if not tasks:
        print(f"ERROR: no tasks found in {args.benchmark_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(tasks)} benchmark tasks")

    # Load model (unless dry-run)
    model = None
    tokenizer = None

    if not args.dry_run:
        adapter_file = args.adapter_dir / "adapters.safetensors"
        if not adapter_file.exists():
            print(f"ERROR: no adapters.safetensors in {args.adapter_dir}", file=sys.stderr)
            return 1

        print(f"Loading base model: {args.model_base} ...")
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers

        model, tokenizer = mlx_load(args.model_base)

        # Load adapter config to determine LoRA/DoRA settings
        config_file = args.adapter_dir / "training_config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            lora_cfg = config.get("lora", {})
            use_dora = lora_cfg.get("use_dora", False)
            adapter_config = {
                "rank": lora_cfg.get("rank", 64),
                "alpha": lora_cfg.get("alpha", 128.0),
                "dropout": lora_cfg.get("dropout", 0.0),
                "scale": lora_cfg.get("alpha", 128.0) / lora_cfg.get("rank", 64),
            }
            if "keys" in lora_cfg:
                adapter_config["keys"] = lora_cfg["keys"]

            linear_to_lora_layers(
                model,
                num_layers=lora_cfg.get("num_layers", 999),
                config=adapter_config,
                use_dora=use_dora,
            )
        else:
            # Fallback: apply default LoRA config
            linear_to_lora_layers(model, num_layers=999, config={"rank": 64, "alpha": 128.0, "scale": 2.0})

        print(f"Loading adapter weights from {args.adapter_dir} ...")
        model.load_weights(str(adapter_file), strict=False)
        print("Model loaded.")
    else:
        print("DRY-RUN mode: using synthetic predictions")

    # Generate predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    n_passed = 0
    n_compiled = 0
    start = time.time()

    with open(args.output, "w") as out_f:
        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{i}")

            if args.dry_run:
                result = generate_dry_run_prediction(task, seed=42)
            else:
                prompt = task.get("prompt", "")
                prediction = generate_prediction_mlx(model, tokenizer, prompt,
                                                     max_tokens=args.max_tokens)
                check = check_with_tkc(prediction)
                result = {
                    "task_id": task_id,
                    "prompt": prompt,
                    "prediction": prediction,
                    "compiled": check["compiled"],
                    "passed": check["passed"],
                    "token_count": count_tokens(prediction),
                    "diagnostics": check["diagnostics"],
                }

            results.append(result)
            out_f.write(json.dumps(result) + "\n")

            if result["passed"]:
                n_passed += 1
            if result["compiled"]:
                n_compiled += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] pass={n_passed} compile={n_compiled}")

    elapsed = time.time() - start
    pass_at_1 = n_passed / len(tasks) if tasks else 0.0
    compile_rate = n_compiled / len(tasks) if tasks else 0.0

    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print(f"  Pass@1:       {pass_at_1:.4f} ({n_passed}/{len(tasks)})")
    print(f"  Compile rate: {compile_rate:.4f} ({n_compiled}/{len(tasks)})")
    print(f"  Output:       {args.output}")

    # Write summary
    summary = {
        "adapter_dir": str(args.adapter_dir),
        "benchmark_dir": str(args.benchmark_dir),
        "n_tasks": len(tasks),
        "pass_at_1": pass_at_1,
        "compile_rate": compile_rate,
        "n_passed": n_passed,
        "n_compiled": n_compiled,
        "elapsed_seconds": elapsed,
        "dry_run": args.dry_run,
    }
    summary_path = args.output.parent / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:      {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
