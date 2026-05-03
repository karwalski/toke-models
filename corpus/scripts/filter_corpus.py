#!/usr/bin/env python3
"""Deduplication and quality filtering pipeline for the toke corpus.

Reads validated JSONL corpus entries, applies exact and near-duplicate
detection, quality filters, and diversity scoring.  Produces filtered
output, duplicate/rejected logs, and a statistics report.

Story 81.2c
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stdlib module categories used for diversity scoring
# ---------------------------------------------------------------------------
STDLIB_CATEGORIES: dict[str, re.Pattern[str]] = {
    "str":   re.compile(r"\bstd\.str\b"),
    "http":  re.compile(r"\bstd\.http\b"),
    "json":  re.compile(r"\bstd\.json\b"),
    "db":    re.compile(r"\bstd\.db\b"),
    "file":  re.compile(r"\bstd\.file\b"),
    "math":  re.compile(r"\bstd\.math\b"),
    "net":   re.compile(r"\bstd\.net\b"),
    "io":    re.compile(r"\bstd\.io\b"),
    "os":    re.compile(r"\bstd\.os\b"),
    "time":  re.compile(r"\bstd\.time\b"),
    "crypto": re.compile(r"\bstd\.crypto\b"),
    "fmt":   re.compile(r"\bstd\.fmt\b"),
}

# Tokenisation pattern: split on whitespace and punctuation boundaries.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|[^\s]")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of filtering a single entry."""

    entry: dict
    kept: bool = True
    duplicate_of: str | None = None
    reject_reasons: list[str] = field(default_factory=list)


@dataclass
class FilterStats:
    """Accumulated statistics for the filtering run."""

    input_count: int = 0
    output_count: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    rejections_by_reason: Counter = field(default_factory=Counter)
    category_distribution: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "total_duplicates": self.exact_duplicates + self.near_duplicates,
            "total_rejected": sum(self.rejections_by_reason.values()),
            "rejections_by_reason": dict(self.rejections_by_reason),
            "category_distribution": dict(self.category_distribution),
            "underrepresented_categories": self._underrepresented(),
        }

    def _underrepresented(self) -> list[str]:
        """Categories with fewer entries than 10% of the median count."""
        if not self.category_distribution:
            return []
        counts = sorted(self.category_distribution.values())
        median = counts[len(counts) // 2] if counts else 0
        threshold = max(1, median * 0.1)
        return [
            cat for cat, count in self.category_distribution.items()
            if count < threshold
        ]


# ---------------------------------------------------------------------------
# Normalisation and tokenisation helpers
# ---------------------------------------------------------------------------

def normalise_source(source: str) -> str:
    """Normalise source for hashing: lowercase, collapse whitespace."""
    text = source.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_source(source: str) -> str:
    """SHA-256 hex digest of normalised source."""
    norm = normalise_source(source)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def tokenise(source: str) -> list[str]:
    """Tokenise source by splitting on whitespace/punctuation boundaries."""
    norm = normalise_source(source)
    return _TOKEN_RE.findall(norm)


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def check_quality(
    source: str,
    *,
    min_lines: int,
    max_lines: int,
    min_tokens: int = 20,
    max_tokens: int = 5000,
) -> list[str]:
    """Run quality filters on a toke source string.

    Returns a list of rejection reasons (empty if the program passes).
    """
    reasons: list[str] = []
    lines = source.strip().splitlines()
    line_count = len(lines)
    token_count = len(tokenise(source))

    # Length bounds.
    if line_count < min_lines:
        reasons.append(f"too_short: {line_count} lines (min {min_lines})")
    if line_count > max_lines:
        reasons.append(f"too_long: {line_count} lines (max {max_lines})")
    if token_count < min_tokens:
        reasons.append(f"too_few_tokens: {token_count} tokens (min {min_tokens})")
    if token_count > max_tokens:
        reasons.append(f"too_many_tokens: {token_count} tokens (max {max_tokens})")

    # Must contain at least one function declaration (f= or F=).
    if not re.search(r"\b[fF]\s*=", source):
        reasons.append("no_function: missing function declaration (f= or F=)")

    # Must contain a module declaration (m= or M=).
    if not re.search(r"\b[mM]\s*=", source):
        reasons.append("no_module: missing module declaration (m= or M=)")

    # No empty function bodies: look for patterns like F=name(...){} with
    # nothing between the braces.
    if re.search(r"\{[\s]*\}", source):
        reasons.append("empty_body: contains empty function body")

    # Not just imports with no logic: if every non-blank line is an import
    # or module declaration and there is no function, reject.
    non_blank = [ln.strip() for ln in lines if ln.strip()]
    import_or_module_lines = [
        ln for ln in non_blank
        if re.match(r"^[iImM]\s*=", ln) or re.match(r"^(import|use)\b", ln, re.IGNORECASE)
    ]
    if len(import_or_module_lines) == len(non_blank):
        reasons.append("imports_only: program is just imports/module declarations with no logic")

    return reasons


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------

def categorise_program(source: str) -> list[str]:
    """Return stdlib categories used by the program."""
    cats: list[str] = []
    for name, pattern in STDLIB_CATEGORIES.items():
        if pattern.search(source):
            cats.append(name)
    if not cats:
        cats.append("none")
    return cats


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_entries(input_dir: Path) -> list[dict]:
    """Load all JSONL entries from the input directory."""
    entries: list[dict] = []
    jsonl_files = sorted(input_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        logger.warning("No .jsonl files found in %s", input_dir)
        return entries

    for filepath in jsonl_files:
        with open(filepath, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d: %s",
                        filepath, line_no, exc,
                    )
    logger.info("Loaded %d entries from %s", len(entries), input_dir)
    return entries


def run_pipeline(
    entries: list[dict],
    *,
    similarity_threshold: float,
    min_lines: int,
    max_lines: int,
    min_tokens: int = 20,
    max_tokens: int = 5000,
) -> tuple[list[dict], list[dict], list[dict], FilterStats]:
    """Run the full deduplication and quality filtering pipeline.

    Returns:
        (filtered, duplicates, rejected, stats)
    """
    stats = FilterStats(input_count=len(entries))

    # --- Phase 1: Exact deduplication ---
    seen_hashes: dict[str, dict] = {}   # hash -> first entry
    exact_dups: list[dict] = []
    unique_entries: list[dict] = []

    for entry in entries:
        source = entry.get("tk_source", "")
        digest = hash_source(source)
        if digest in seen_hashes:
            stats.exact_duplicates += 1
            dup_record = dict(entry)
            dup_record["_duplicate_of"] = seen_hashes[digest].get("id", "unknown")
            dup_record["_duplicate_type"] = "exact"
            exact_dups.append(dup_record)
        else:
            seen_hashes[digest] = entry
            unique_entries.append(entry)

    logger.info(
        "Exact dedup: %d -> %d (removed %d)",
        len(entries), len(unique_entries), stats.exact_duplicates,
    )

    # --- Phase 2: Near-duplicate detection (token-level Jaccard) ---
    # Pre-compute token sets for all unique entries.
    token_sets: list[tuple[dict, set[str]]] = []
    for entry in unique_entries:
        source = entry.get("tk_source", "")
        tokens = set(tokenise(source))
        token_sets.append((entry, tokens))

    near_dups: list[dict] = []
    kept_after_near: list[dict] = []
    removed_indices: set[int] = set()

    for i in range(len(token_sets)):
        if i in removed_indices:
            continue
        entry_a, tokens_a = token_sets[i]
        for j in range(i + 1, len(token_sets)):
            if j in removed_indices:
                continue
            entry_b, tokens_b = token_sets[j]
            sim = jaccard_similarity(tokens_a, tokens_b)
            if sim >= similarity_threshold:
                # Keep the longer program.
                source_a = entry_a.get("tk_source", "")
                source_b = entry_b.get("tk_source", "")
                if len(source_b) > len(source_a):
                    # Remove a, keep b.
                    removed_indices.add(i)
                    stats.near_duplicates += 1
                    dup_record = dict(entry_a)
                    dup_record["_duplicate_of"] = entry_b.get("id", "unknown")
                    dup_record["_duplicate_type"] = "near"
                    dup_record["_similarity"] = round(sim, 4)
                    near_dups.append(dup_record)
                    break  # entry_a is removed, stop comparing it
                else:
                    # Remove b, keep a.
                    removed_indices.add(j)
                    stats.near_duplicates += 1
                    dup_record = dict(entry_b)
                    dup_record["_duplicate_of"] = entry_a.get("id", "unknown")
                    dup_record["_duplicate_type"] = "near"
                    dup_record["_similarity"] = round(sim, 4)
                    near_dups.append(dup_record)

    for i, (entry, _) in enumerate(token_sets):
        if i not in removed_indices:
            kept_after_near.append(entry)

    logger.info(
        "Near dedup: %d -> %d (removed %d, threshold=%.2f)",
        len(unique_entries), len(kept_after_near),
        stats.near_duplicates, similarity_threshold,
    )

    all_dups = exact_dups + near_dups

    # --- Phase 3: Quality filtering ---
    filtered: list[dict] = []
    rejected: list[dict] = []

    for entry in kept_after_near:
        source = entry.get("tk_source", "")
        reasons = check_quality(
            source,
            min_lines=min_lines,
            max_lines=max_lines,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        if reasons:
            reject_record = dict(entry)
            reject_record["_reject_reasons"] = reasons
            rejected.append(reject_record)
            for reason in reasons:
                # Use the tag before the colon as the reason key.
                tag = reason.split(":")[0]
                stats.rejections_by_reason[tag] += 1
        else:
            filtered.append(entry)

    logger.info(
        "Quality filter: %d -> %d (rejected %d)",
        len(kept_after_near), len(filtered), len(rejected),
    )

    # --- Phase 4: Diversity scoring ---
    for entry in filtered:
        source = entry.get("tk_source", "")
        cats = categorise_program(source)
        for cat in cats:
            stats.category_distribution[cat] += 1

    stats.output_count = len(filtered)

    return filtered, all_dups, rejected, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(entries: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


def write_report(stats: FilterStats, path: Path) -> None:
    """Write the filter report as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logger.info("Wrote filter report to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate and quality-filter the toke corpus.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing validated .jsonl corpus files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write filtered output files.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Jaccard similarity threshold for near-duplicate detection (default: 0.9).",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=3,
        help="Minimum number of lines for a valid program (default: 3).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Maximum number of lines for a valid program (default: 500).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=20,
        help="Minimum number of tokens for a valid program (default: 20).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum number of tokens for a valid program (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    entries = load_entries(input_dir)
    if not entries:
        logger.error("No entries loaded. Nothing to filter.")
        return 1

    filtered, duplicates, rejected, stats = run_pipeline(
        entries,
        similarity_threshold=args.similarity_threshold,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    # Print summary.
    report = stats.to_dict()
    print("\n--- Filter Report ---")
    print(f"  Input:              {report['input_count']}")
    print(f"  Output:             {report['output_count']}")
    print(f"  Exact duplicates:   {report['exact_duplicates']}")
    print(f"  Near duplicates:    {report['near_duplicates']}")
    print(f"  Total rejected:     {report['total_rejected']}")
    if report["rejections_by_reason"]:
        print("  Rejections by reason:")
        for reason, count in sorted(report["rejections_by_reason"].items()):
            print(f"    {reason}: {count}")
    if report["category_distribution"]:
        print("  Category distribution:")
        for cat, count in sorted(
            report["category_distribution"].items(), key=lambda x: -x[1]
        ):
            print(f"    {cat}: {count}")
    if report["underrepresented_categories"]:
        print(f"  Underrepresented:   {', '.join(report['underrepresented_categories'])}")
    print("---")

    if args.dry_run:
        logger.info("Dry run — no output files written.")
        return 0

    # Write output files.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(filtered, output_dir / "filtered.jsonl")
    write_jsonl(duplicates, output_dir / "duplicates.jsonl")
    write_jsonl(rejected, output_dir / "rejected.jsonl")
    write_report(stats, output_dir / "filter_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
