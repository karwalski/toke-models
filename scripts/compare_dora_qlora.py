#!/usr/bin/env python3
"""Compare DoRA vs QLoRA evaluation results.

Loads evaluation results from two adapter directories, computes comparison
metrics, runs statistical significance tests, and outputs a report in both
JSON and markdown table format.

Usage:
    python scripts/compare_dora_qlora.py \
        --qlora-dir adapters/qlora/ \
        --dora-dir adapters/dora/ \
        --benchmark-dir eval/benchmark/ \
        --output-dir results/comparison/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_predictions(adapter_dir: Path) -> list[dict]:
    """Load predictions JSONL from an adapter's evaluation output."""
    predictions_file = adapter_dir / "predictions.jsonl"
    if not predictions_file.exists():
        raise FileNotFoundError(f"No predictions.jsonl in {adapter_dir}")
    results = []
    with open(predictions_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_training_summary(adapter_dir: Path) -> dict | None:
    """Load training_summary.json if available."""
    summary_file = adapter_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def load_training_log(adapter_dir: Path) -> list[dict] | None:
    """Load training_log.jsonl for loss curve comparison."""
    log_file = adapter_dir / "training_log.jsonl"
    if not log_file.exists():
        return None
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_pass_at_1(predictions: list[dict]) -> float:
    """Compute Pass@1 from prediction results."""
    if not predictions:
        return 0.0
    passed = sum(1 for p in predictions if p.get("passed", False))
    return passed / len(predictions)


def compute_compile_rate(predictions: list[dict]) -> float:
    """Compute compile rate from prediction results."""
    if not predictions:
        return 0.0
    compiled = sum(1 for p in predictions if p.get("compiled", False))
    return compiled / len(predictions)


def compute_avg_token_count(predictions: list[dict]) -> float:
    """Compute average token count from prediction results."""
    counts = [p["token_count"] for p in predictions if "token_count" in p]
    if not counts:
        return 0.0
    return float(np.mean(counts))


def mcnemar_test(preds_a: list[dict], preds_b: list[dict]) -> dict:
    """McNemar's test for paired nominal data.

    Compares pass/fail outcomes between two models on the same tasks.
    Returns test statistic, p-value, and contingency counts.
    """
    # Build task_id -> passed maps
    a_by_task = {p["task_id"]: p.get("passed", False) for p in preds_a if "task_id" in p}
    b_by_task = {p["task_id"]: p.get("passed", False) for p in preds_b if "task_id" in p}

    common_tasks = sorted(set(a_by_task) & set(b_by_task))
    if not common_tasks:
        return {"error": "no overlapping tasks", "p_value": 1.0}

    # Contingency: b=both pass, c=A pass only, d=B pass only, e=both fail
    b_count = sum(1 for t in common_tasks if a_by_task[t] and b_by_task[t])
    c_count = sum(1 for t in common_tasks if a_by_task[t] and not b_by_task[t])
    d_count = sum(1 for t in common_tasks if not a_by_task[t] and b_by_task[t])
    e_count = sum(1 for t in common_tasks if not a_by_task[t] and not b_by_task[t])

    n_discordant = c_count + d_count
    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "both_pass": b_count,
            "qlora_only": c_count,
            "dora_only": d_count,
            "both_fail": e_count,
            "n_tasks": len(common_tasks),
            "note": "no discordant pairs",
        }

    # McNemar's test with continuity correction
    statistic = (abs(c_count - d_count) - 1) ** 2 / (c_count + d_count)
    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "both_pass": b_count,
        "qlora_only": c_count,
        "dora_only": d_count,
        "both_fail": e_count,
        "n_tasks": len(common_tasks),
    }


def paired_ttest_token_count(preds_a: list[dict], preds_b: list[dict]) -> dict:
    """Paired t-test on token counts for the same tasks."""
    a_by_task = {p["task_id"]: p.get("token_count", 0) for p in preds_a if "task_id" in p}
    b_by_task = {p["task_id"]: p.get("token_count", 0) for p in preds_b if "task_id" in p}

    common_tasks = sorted(set(a_by_task) & set(b_by_task))
    if len(common_tasks) < 2:
        return {"error": "insufficient overlapping tasks", "p_value": 1.0}

    a_vals = np.array([a_by_task[t] for t in common_tasks], dtype=float)
    b_vals = np.array([b_by_task[t] for t in common_tasks], dtype=float)

    stat, p_val = stats.ttest_rel(a_vals, b_vals)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "mean_qlora": float(np.mean(a_vals)),
        "mean_dora": float(np.mean(b_vals)),
        "n_tasks": len(common_tasks),
    }


def compare_loss_curves(qlora_log: list[dict] | None, dora_log: list[dict] | None) -> dict:
    """Compare final training losses from log files."""
    result = {}
    for label, log in [("qlora", qlora_log), ("dora", dora_log)]:
        if log:
            losses = [e["loss"] for e in log if "loss" in e]
            if losses:
                result[f"{label}_final_loss"] = losses[-1]
                result[f"{label}_min_loss"] = min(losses)
                result[f"{label}_n_steps"] = len(losses)
    return result


def build_comparison(
    qlora_preds: list[dict],
    dora_preds: list[dict],
    qlora_summary: dict | None,
    dora_summary: dict | None,
    qlora_log: list[dict] | None,
    dora_log: list[dict] | None,
) -> dict:
    """Build full comparison report."""
    qlora_pass1 = compute_pass_at_1(qlora_preds)
    dora_pass1 = compute_pass_at_1(dora_preds)
    qlora_compile = compute_compile_rate(qlora_preds)
    dora_compile = compute_compile_rate(dora_preds)
    qlora_tokens = compute_avg_token_count(qlora_preds)
    dora_tokens = compute_avg_token_count(dora_preds)

    report = {
        "metrics": {
            "qlora": {
                "pass_at_1": qlora_pass1,
                "compile_rate": qlora_compile,
                "avg_token_count": qlora_tokens,
                "n_predictions": len(qlora_preds),
            },
            "dora": {
                "pass_at_1": dora_pass1,
                "compile_rate": dora_compile,
                "avg_token_count": dora_tokens,
                "n_predictions": len(dora_preds),
            },
            "delta": {
                "pass_at_1": dora_pass1 - qlora_pass1,
                "compile_rate": dora_compile - qlora_compile,
                "avg_token_count": dora_tokens - qlora_tokens,
            },
        },
        "significance": {
            "mcnemar_pass_at_1": mcnemar_test(qlora_preds, dora_preds),
            "paired_ttest_token_count": paired_ttest_token_count(qlora_preds, dora_preds),
        },
        "loss_curves": compare_loss_curves(qlora_log, dora_log),
    }

    # Include training summaries if available
    if qlora_summary:
        report["qlora_training"] = {
            k: qlora_summary[k]
            for k in ["adapter_type", "lora_rank", "lora_alpha", "epochs",
                       "training_time_seconds", "trainable_parameters"]
            if k in qlora_summary
        }
    if dora_summary:
        report["dora_training"] = {
            k: dora_summary[k]
            for k in ["adapter_type", "lora_rank", "lora_alpha", "epochs",
                       "training_time_seconds", "trainable_parameters"]
            if k in dora_summary
        }

    return report


def format_markdown(report: dict) -> str:
    """Format comparison report as a markdown table."""
    m = report["metrics"]
    lines = [
        "# DoRA vs QLoRA Comparison Report",
        "",
        "## Metrics",
        "",
        "| Metric | QLoRA | DoRA | Delta |",
        "|--------|-------|------|-------|",
        f"| Pass@1 | {m['qlora']['pass_at_1']:.4f} | {m['dora']['pass_at_1']:.4f} | {m['delta']['pass_at_1']:+.4f} |",
        f"| Compile Rate | {m['qlora']['compile_rate']:.4f} | {m['dora']['compile_rate']:.4f} | {m['delta']['compile_rate']:+.4f} |",
        f"| Avg Token Count | {m['qlora']['avg_token_count']:.1f} | {m['dora']['avg_token_count']:.1f} | {m['delta']['avg_token_count']:+.1f} |",
        f"| N Predictions | {m['qlora']['n_predictions']} | {m['dora']['n_predictions']} | — |",
        "",
    ]

    # Significance tests
    sig = report["significance"]
    mcn = sig["mcnemar_pass_at_1"]
    ttest = sig["paired_ttest_token_count"]

    lines.extend([
        "## Statistical Significance",
        "",
        "### McNemar's Test (Pass@1)",
        "",
    ])

    if "error" not in mcn:
        lines.extend([
            f"- Chi-squared statistic: {mcn['statistic']:.4f}",
            f"- p-value: {mcn['p_value']:.6f}",
            f"- Both pass: {mcn['both_pass']}, QLoRA only: {mcn['qlora_only']}, "
            f"DoRA only: {mcn['dora_only']}, Both fail: {mcn['both_fail']}",
            f"- N tasks: {mcn['n_tasks']}",
            f"- Significant (p<0.05): {'Yes' if mcn['p_value'] < 0.05 else 'No'}",
        ])
    else:
        lines.append(f"- {mcn['error']}")

    lines.extend(["", "### Paired t-test (Token Count)", ""])

    if "error" not in ttest:
        lines.extend([
            f"- t-statistic: {ttest['statistic']:.4f}",
            f"- p-value: {ttest['p_value']:.6f}",
            f"- Mean QLoRA: {ttest['mean_qlora']:.1f}, Mean DoRA: {ttest['mean_dora']:.1f}",
            f"- N tasks: {ttest['n_tasks']}",
            f"- Significant (p<0.05): {'Yes' if ttest['p_value'] < 0.05 else 'No'}",
        ])
    else:
        lines.append(f"- {ttest['error']}")

    # Loss curves
    lc = report.get("loss_curves", {})
    if lc:
        lines.extend(["", "## Training Loss", ""])
        for key in ["qlora_final_loss", "dora_final_loss", "qlora_min_loss", "dora_min_loss"]:
            if key in lc:
                lines.append(f"- {key}: {lc[key]:.6f}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qlora-dir", required=True, type=Path,
                        help="Path to QLoRA adapter/evaluation directory")
    parser.add_argument("--dora-dir", required=True, type=Path,
                        help="Path to DoRA adapter/evaluation directory")
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Path to benchmark tasks (for reference, not currently used)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for comparison output files")
    args = parser.parse_args(argv)

    # Validate inputs
    for label, d in [("QLoRA", args.qlora_dir), ("DoRA", args.dora_dir)]:
        if not d.exists():
            print(f"ERROR: {label} directory not found: {d}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading QLoRA predictions from {args.qlora_dir} ...")
    qlora_preds = load_predictions(args.qlora_dir)
    print(f"  {len(qlora_preds)} predictions")

    print(f"Loading DoRA predictions from {args.dora_dir} ...")
    dora_preds = load_predictions(args.dora_dir)
    print(f"  {len(dora_preds)} predictions")

    qlora_summary = load_training_summary(args.qlora_dir)
    dora_summary = load_training_summary(args.dora_dir)
    qlora_log = load_training_log(args.qlora_dir)
    dora_log = load_training_log(args.dora_dir)

    # Build comparison
    print("Computing comparison metrics ...")
    report = build_comparison(qlora_preds, dora_preds,
                              qlora_summary, dora_summary,
                              qlora_log, dora_log)

    # Write JSON
    json_path = args.output_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report: {json_path}")

    # Write markdown
    md_path = args.output_dir / "comparison_report.md"
    md_content = format_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown report: {md_path}")

    # Print summary
    m = report["metrics"]
    print(f"\n{'='*50}")
    print(f"Pass@1  QLoRA={m['qlora']['pass_at_1']:.4f}  DoRA={m['dora']['pass_at_1']:.4f}  delta={m['delta']['pass_at_1']:+.4f}")
    print(f"Compile QLoRA={m['qlora']['compile_rate']:.4f}  DoRA={m['dora']['compile_rate']:.4f}  delta={m['delta']['compile_rate']:+.4f}")
    mcn_p = report["significance"]["mcnemar_pass_at_1"].get("p_value", 1.0)
    print(f"McNemar p={mcn_p:.6f} {'*' if mcn_p < 0.05 else ''}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
