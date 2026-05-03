#!/usr/bin/env python3
"""Transform legacy toke corpus to current 56-char default syntax.

SUPERSEDED: This Python script has been superseded by `toke --migrate` which
handles ALL the transforms below (and more) natively in the C compiler.
Use migrate_corpus.sh for new corpus generation pipelines.

This script is retained as a fallback for edge cases where the C tool cannot
process a file (e.g., malformed input that crashes the parser). The primary
pipeline should be:

    input .tk files -> toke --migrate -> toke --check -> validated output

See migrate_corpus.sh for the canonical pipeline.

Story 81.2a — Legacy-to-default syntax transformer (Python fallback).

Transformations (applied in order):
  1. Uppercase type names -> $sigil       User -> $user, Err -> $err
  2. Square brackets -> @()               [1,2,3] -> @(1;2;3), a[i] -> a.get(i)
  3. Match syntax                         expr|{ -> mt expr {
  4. Underscore removal                   to_int -> toint
  5. Comment stripping                    (* ... *) removed
  6. Semicolon separators                 , in function args -> ;
  7. Return shorthand                     return x / ret x -> <x

Usage:
  python transform_corpus.py --input corpus/legacy/ --output corpus/default/
  python transform_corpus.py --input corpus/legacy.jsonl --output corpus/default.jsonl
  python transform_corpus.py --input corpus/legacy/ --output corpus/default/ --toke ~/tk/toke/toke
  python transform_corpus.py --input corpus/legacy/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primitive / built-in types that are already lowercase and must NOT get a $ sigil.
# Note: "Str" (uppercase) IS transformed to "$str" — only lowercase forms are exempt.
PRIMITIVE_TYPES = frozenset({
    "i64", "u64", "f64", "bool", "void",
})

# toke keywords — must never be treated as identifiers for underscore removal
KEYWORDS = frozenset({
    "if", "el", "lp", "br", "let", "mut", "as", "true", "false",
    "mt", "fn", "ret", "return", "mod", "use", "pub",
})

# Legacy declaration keyword patterns (uppercase single-char before =)
DECL_UPPERCASE = frozenset({"M", "F", "I", "T", "C"})


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

@dataclass
class TransformStats:
    total: int = 0
    transformed: int = 0
    skipped: int = 0
    validated: int = 0
    passed: int = 0
    failed: int = 0
    type_sigils: int = 0
    bracket_to_at: int = 0
    index_to_get: int = 0
    match_rewrites: int = 0
    underscore_removals: int = 0
    comments_stripped: int = 0
    comma_to_semi: int = 0
    return_shorthand: int = 0

    def summary(self) -> str:
        lines = [
            f"Total programs:        {self.total}",
            f"Transformed:           {self.transformed}",
            f"Skipped (unclean):     {self.skipped}",
        ]
        if self.validated > 0:
            lines.append(f"Validated:             {self.validated}")
            lines.append(f"  Passed:              {self.passed}")
            lines.append(f"  Failed:              {self.failed}")
            if self.validated > 0:
                rate = self.passed / self.validated * 100
                lines.append(f"  Pass rate:           {rate:.1f}%")
        lines.append("")
        lines.append("Transformation counts:")
        lines.append(f"  Type sigils ($):     {self.type_sigils}")
        lines.append(f"  [] -> @():           {self.bracket_to_at}")
        lines.append(f"  a[i] -> a.get(i):    {self.index_to_get}")
        lines.append(f"  |{{ -> mt:            {self.match_rewrites}")
        lines.append(f"  Underscores removed: {self.underscore_removals}")
        lines.append(f"  Comments stripped:    {self.comments_stripped}")
        lines.append(f"  , -> ; in args:      {self.comma_to_semi}")
        lines.append(f"  return/ret -> <:     {self.return_shorthand}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual transformation passes (regex-based, conservative)
# ---------------------------------------------------------------------------

def strip_comments(source: str, stats: TransformStats) -> str:
    """Remove all (* ... *) comment blocks.

    Handles nested comments with a stack-based scanner.
    Does NOT remove comments inside string literals.
    """
    # Protect string literals by replacing them with placeholders
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    # Stack-based nested comment removal
    result = []
    i = 0
    n = len(work)
    depth = 0

    while i < n:
        if i + 1 < n and work[i] == '(' and work[i + 1] == '*':
            if depth == 0:
                stats.comments_stripped += 1
            depth += 1
            i += 2
        elif i + 1 < n and work[i] == '*' and work[i + 1] == ')':
            depth -= 1
            i += 2
            if depth < 0:
                depth = 0  # malformed, recover
        elif depth == 0:
            result.append(work[i])
            i += 1
        else:
            i += 1  # inside comment, skip

    work = ''.join(result)

    # Restore string literals
    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_type_sigils(source: str, stats: TransformStats) -> str:
    """Uppercase type names -> $sigil.

    User -> $user, Err -> $err, Ok -> $ok, Str -> $str (but not i64, bool, etc.)
    Only transforms identifiers that start with an uppercase letter and are
    NOT inside string literals.
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    # Transform uppercase-initial identifiers that look like type names.
    # We are conservative: only transform if the identifier is preceded by
    # a type-position indicator (:, !, =after t/T) or is followed by { (struct literal).
    # Also handle declaration keywords M=, F=, T=, etc.
    def _sigil_replace(m: re.Match) -> str:
        name = m.group(0)
        # Only skip if the name is already lowercase AND a primitive
        if name in PRIMITIVE_TYPES:
            return name  # leave lowercase primitives alone
        if name in DECL_UPPERCASE:
            return name  # handled separately
        if name in KEYWORDS:
            return name
        stats.type_sigils += 1
        return "$" + name.lower()

    # Type positions: after :, after !, in T=Name patterns, before { for literals
    # Pattern: uppercase-initial identifier NOT preceded by $ (already sigilled)
    # and NOT inside a string
    work = re.sub(
        r'(?<![$a-zA-Z])([A-Z][a-zA-Z0-9]*)',
        lambda m: _sigil_replace(m) if m.group(1) not in DECL_UPPERCASE else m.group(0),
        work,
    )

    # Lowercase the single-char declaration keywords: M= -> m=, F= -> f=, etc.
    def _lower_decl(m: re.Match) -> str:
        return m.group(1).lower() + "="
    work = re.sub(r'\b([MFITC])=', _lower_decl, work)

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_brackets(source: str, stats: TransformStats) -> str:
    """Square brackets -> @() for array literals, a[i] -> a.get(i).

    This is the trickiest transformation. We distinguish:
    - Array literals: [1,2,3] or [1;2;3] -> @(1;2;3)
    - Array types: [i64] after : -> @i64 (but this overlaps with literals)
    - Indexing: ident[expr] -> ident.get(expr)  (variable index)
                ident[N]    -> ident.N           (constant index)

    Conservative approach: use regex to find [ and classify by context.
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    result = []
    i = 0
    n = len(work)

    while i < n:
        # Find next [
        if work[i] != '[':
            result.append(work[i])
            i += 1
            continue

        # Find matching ]
        bracket_start = i
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if work[j] == '[':
                depth += 1
            elif work[j] == ']':
                depth -= 1
            j += 1
        # j is now past the closing ]
        bracket_end = j
        inner = work[i + 1:j - 1]

        # Determine context: what precedes the [?
        pre = ''.join(result).rstrip()

        # Indexing context: preceded by identifier, ), or ]
        is_indexing = bool(re.search(r'[a-zA-Z0-9_\)\]]$', pre))

        if is_indexing:
            # Is the index a simple integer constant?
            stripped_inner = inner.strip()
            if re.match(r'^\d+$', stripped_inner):
                # ident[N] -> ident.N
                stats.index_to_get += 1
                result.append('.')
                result.append(stripped_inner)
            else:
                # ident[expr] -> ident.get(expr)
                stats.index_to_get += 1
                result.append('.get(')
                result.append(inner)
                result.append(')')
        else:
            # Array literal or type: [ ... ] -> @( ... )
            stats.bracket_to_at += 1
            result.append('@(')
            result.append(inner)
            result.append(')')

        i = bracket_end

    work = ''.join(result)

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_match_syntax(source: str, stats: TransformStats) -> str:
    """Match syntax: expr|{ -> mt expr {

    Legacy: result|{Ok:v do_stuff(v);Err:e handle(e)}
    Default: mt result {$ok:v do_stuff(v);$err:e handle(e)}

    We look for the pattern: <expr>|{ and rewrite to: mt <expr> {
    The expr is the token(s) immediately preceding |{.
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    # Pattern: identifier or closing paren/bracket followed by |{
    # The expression can be: a single ident, a function call like f(), etc.
    # Conservative: capture the immediately preceding simple expression.
    def _rewrite_match(m: re.Match) -> str:
        expr = m.group(1)
        stats.match_rewrites += 1
        return f"mt {expr} {{"

    # Match: word-char or ) or ] followed by |{
    # Capture the preceding expression — greedily grab the simple expression
    # Simple expression: identifier, or ident(...), or chained calls
    work = re.sub(
        r'(\b[a-zA-Z_$][a-zA-Z0-9_$.]*(?:\([^)]*\))?)\|\{',
        _rewrite_match,
        work,
    )

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_underscores(source: str, stats: TransformStats) -> str:
    """Remove underscores from identifiers: to_int -> toint, from_bytes -> frombytes.

    Only transforms identifiers (not strings, not comments which are already stripped).
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    def _remove_underscores(m: re.Match) -> str:
        ident = m.group(0)
        if '_' not in ident:
            return ident
        stats.underscore_removals += 1
        return ident.replace('_', '')

    # Match identifiers that contain underscores
    work = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', _remove_underscores, work)

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_comma_to_semicolon(source: str, stats: TransformStats) -> str:
    """Replace , with ; in function argument lists.

    Conservative: only replace commas inside parenthesised argument lists,
    not inside strings or other contexts.
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    # Replace , with ; when inside parentheses (function args)
    # Track paren depth to only replace at appropriate levels
    result = []
    paren_depth = 0
    for ch in work:
        if ch == '(':
            paren_depth += 1
            result.append(ch)
        elif ch == ')':
            paren_depth -= 1
            result.append(ch)
        elif ch == ',' and paren_depth > 0:
            stats.comma_to_semi += 1
            result.append(';')
        else:
            result.append(ch)

    work = ''.join(result)

    # Also replace commas in @() array literals (which were [] before)
    # These may use , as separators in legacy code
    # After bracket transform, they are @(...) so commas inside @() should be ;
    def _semi_in_at(m: re.Match) -> str:
        inner = m.group(1)
        count = inner.count(',')
        if count > 0:
            stats.comma_to_semi += count
        return '@(' + inner.replace(',', ';') + ')'

    work = re.sub(r'@\(([^)]*)\)', _semi_in_at, work)

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


def transform_return_shorthand(source: str, stats: TransformStats) -> str:
    """return x / ret x -> <x.

    Handles both 'return' and 'ret' keywords.
    """
    strings: list[str] = []
    def _save_string(m: re.Match) -> str:
        strings.append(m.group(0))
        return f"\x00xqs{len(strings) - 1}\x00"

    work = re.sub(r'"(?:[^"\\]|\\.)*"', _save_string, source)

    def _rewrite_return(m: re.Match) -> str:
        stats.return_shorthand += 1
        expr = m.group(1)
        return f"<{expr}"

    # Match 'return expr' or 'ret expr' — expr goes until ; or } or end
    work = re.sub(r'\breturn\s+([^;}\n]+)', _rewrite_return, work)
    work = re.sub(r'\bret\s+([^;}\n]+)', _rewrite_return, work)

    def _restore_string(m: re.Match) -> str:
        idx = int(m.group(1))
        return strings[idx]

    work = re.sub(r'\x00xqs(\d+)\x00', _restore_string, work)
    return work


# ---------------------------------------------------------------------------
# Native migration via `toke --migrate`
# ---------------------------------------------------------------------------

# Default path to the toke binary; overridden by --toke CLI flag.
_TOKE_BINARY: Optional[str] = None


def migrate_with_toke(source: str, toke_path: str) -> tuple[str, bool]:
    """Run `toke --migrate` on source via a temp file.

    Returns (migrated_source, success).  Falls back to (source, False) on any
    error so the caller can retry with the Python pipeline.
    """
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tk", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        proc = subprocess.run(
            [toke_path, "--migrate", tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout, True
        return source, False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return source, False
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


# ---------------------------------------------------------------------------
# Main transformation pipeline
# ---------------------------------------------------------------------------

def transform_program(source: str, stats: TransformStats) -> tuple[str, bool]:
    """Apply all transformations to a single program.

    Primary path: delegate to `toke --migrate` (native C tool).
    Fallback: apply Python regex transforms if the native tool is unavailable
    or fails on this input.

    Returns (transformed_source, success).
    If the transformation produces something that looks broken (unbalanced
    brackets, etc.), returns the original with success=False.
    """
    # ---- Primary path: native toke --migrate ----
    if _TOKE_BINARY:
        migrated, ok = migrate_with_toke(source, _TOKE_BINARY)
        if ok:
            stats.transformed += 0  # counted by caller
            return migrated, True
        # Fall through to Python fallback

    # ---- Fallback: Python regex transforms ----
    try:
        result = source

        # Order matters — follow the documented sequence
        # 5. Comment stripping (do early so comments don't confuse other passes)
        result = strip_comments(result, stats)

        # 1. Uppercase type names -> $sigil
        result = transform_type_sigils(result, stats)

        # 2. Square brackets -> @()
        result = transform_brackets(result, stats)

        # 3. Match syntax: expr|{ -> mt expr {
        result = transform_match_syntax(result, stats)

        # 6. Semicolon separators: , -> ; in function args
        result = transform_comma_to_semicolon(result, stats)

        # 4. Underscore removal (after other transforms so we don't break patterns)
        result = transform_underscores(result, stats)

        # 7. Return shorthand: return x / ret x -> <x
        result = transform_return_shorthand(result, stats)

        # Sanity checks
        if not _sanity_check(result):
            return source, False

        return result, True

    except Exception:
        return source, False


def _sanity_check(source: str) -> bool:
    """Basic sanity checks on transformed output."""
    # Check balanced braces
    depth = 0
    for ch in source:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False

    # Check balanced parens
    depth = 0
    for ch in source:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        return False

    # No remaining square brackets (should all be transformed)
    # Allow [ and ] inside string literals
    in_string = False
    for i, ch in enumerate(source):
        if ch == '"' and (i == 0 or source[i - 1] != '\\'):
            in_string = not in_string
        if not in_string and ch in '[]':
            return False

    return True


# ---------------------------------------------------------------------------
# Compilation validation
# ---------------------------------------------------------------------------

def validate_with_toke(source: str, toke_path: str) -> tuple[bool, str]:
    """Run toke --check --diag-json on a program via a temp file.

    Returns (passed, diagnostic_output).
    """
    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tk", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        proc = subprocess.run(
            [toke_path, "--check", "--diag-json", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = proc.stdout + proc.stderr
        return proc.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except (FileNotFoundError, OSError) as e:
        return False, str(e)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# File I/O: .tk files in a directory
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    output_dir: Optional[Path],
    toke_path: Optional[str],
    dry_run: bool,
    stats: TransformStats,
) -> list[dict]:
    """Process a directory of .tk files. Returns list of failure records."""
    failures: list[dict] = []

    tk_files = sorted(input_dir.rglob("*.tk"))
    if not tk_files:
        print(f"Warning: no .tk files found in {input_dir}", file=sys.stderr)
        return failures

    for tk_file in tk_files:
        stats.total += 1
        source = tk_file.read_text(encoding="utf-8", errors="replace")

        transformed, success = transform_program(source, stats)

        if not success:
            stats.skipped += 1
            failures.append({
                "file": str(tk_file),
                "reason": "transformation_failed",
                "original": source[:500],
            })
            continue

        # Validate with toke --check if available
        if toke_path:
            stats.validated += 1
            passed, diag = validate_with_toke(transformed, toke_path)
            if passed:
                stats.passed += 1
            else:
                stats.failed += 1
                failures.append({
                    "file": str(tk_file),
                    "reason": "tkc_check_failed",
                    "diagnostic": diag,
                    "transformed": transformed[:500],
                })
                # Still write the file — the caller decides what to do with failures
                # But mark it so the report is clear

        stats.transformed += 1

        if dry_run:
            if stats.total <= 20:
                print(f"--- {tk_file.name} ---")
                print(f"  BEFORE: {source[:200]}")
                print(f"  AFTER:  {transformed[:200]}")
                print()
        elif output_dir is not None:
            rel = tk_file.relative_to(input_dir)
            out_file = output_dir / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(transformed, encoding="utf-8")

    return failures


# ---------------------------------------------------------------------------
# File I/O: JSONL
# ---------------------------------------------------------------------------

def process_jsonl(
    input_path: Path,
    output_path: Optional[Path],
    toke_path: Optional[str],
    dry_run: bool,
    stats: TransformStats,
) -> list[dict]:
    """Process a JSONL file with a 'code' field. Returns list of failure records."""
    failures: list[dict] = []
    output_entries: list[str] = []

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats.total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                stats.skipped += 1
                failures.append({
                    "line": lineno,
                    "reason": f"json_parse_error: {e}",
                })
                continue

            # Support both "code" and "tk_source" field names
            code_field = "code" if "code" in entry else "tk_source"
            source = entry.get(code_field, "")
            if not source:
                stats.skipped += 1
                failures.append({
                    "line": lineno,
                    "reason": "empty_source",
                    "id": entry.get("id", "?"),
                })
                continue

            transformed, success = transform_program(source, stats)

            if not success:
                stats.skipped += 1
                failures.append({
                    "line": lineno,
                    "reason": "transformation_failed",
                    "id": entry.get("id", "?"),
                    "original": source[:500],
                })
                continue

            if toke_path:
                stats.validated += 1
                passed, diag = validate_with_toke(transformed, toke_path)
                if passed:
                    stats.passed += 1
                else:
                    stats.failed += 1
                    failures.append({
                        "line": lineno,
                        "reason": "tkc_check_failed",
                        "id": entry.get("id", "?"),
                        "diagnostic": diag,
                        "transformed": transformed[:500],
                    })

            stats.transformed += 1
            entry[code_field] = transformed
            output_entries.append(json.dumps(entry, ensure_ascii=False))

            if dry_run and stats.total <= 20:
                print(f"--- Line {lineno} (id={entry.get('id', '?')}) ---")
                print(f"  BEFORE: {source[:200]}")
                print(f"  AFTER:  {transformed[:200]}")
                print()

    if not dry_run and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry_line in output_entries:
                f.write(entry_line + "\n")

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transform legacy toke corpus to current 55-char default syntax (story 81.2a)",
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory of .tk files, or a .jsonl file",
    )
    parser.add_argument(
        "--output",
        help="Output directory or .jsonl file (default: <input>_default/)",
    )
    parser.add_argument(
        "--toke",
        help="Path to toke binary for --migrate and --check (default: ~/tk/toke/toke)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing output",
    )
    parser.add_argument(
        "--failures",
        help="Path to write failure records (JSONL). Default: <output>/failures.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    stats = TransformStats()

    # Determine input type and default output
    is_jsonl = input_path.suffix in (".jsonl", ".json") and input_path.is_file()

    if args.output:
        output_path = Path(args.output)
    elif is_jsonl:
        output_path = input_path.with_stem(input_path.stem + "_default")
    else:
        output_path = Path(str(input_path).rstrip("/") + "_default")

    # Resolve toke binary — used for both --migrate (primary) and --check (validation)
    global _TOKE_BINARY
    toke_path = args.toke or os.path.expanduser("~/tk/toke/toke")
    if os.path.isfile(toke_path):
        _TOKE_BINARY = toke_path
        print(f"toke:   {toke_path} (native --migrate enabled)")
    else:
        _TOKE_BINARY = None
        print(f"toke:   not found at {toke_path} (Python fallback only)", file=sys.stderr)

    # Determine failures output path
    if args.failures:
        failures_path = Path(args.failures)
    elif is_jsonl:
        failures_path = output_path.with_stem(output_path.stem + "_failures")
    else:
        failures_path = output_path / "failures.jsonl"

    # Run transformation
    print(f"Input:  {input_path}")
    if not args.dry_run:
        print(f"Output: {output_path}")
    print()

    if is_jsonl:
        failures = process_jsonl(
            input_path,
            output_path if not args.dry_run else None,
            _TOKE_BINARY,
            args.dry_run,
            stats,
        )
    elif input_path.is_dir():
        failures = process_directory(
            input_path,
            output_path if not args.dry_run else None,
            _TOKE_BINARY,
            args.dry_run,
            stats,
        )
    else:
        print(f"Error: {input_path} is not a directory or JSONL file", file=sys.stderr)
        return 1

    # Write failures
    if failures and not args.dry_run:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w", encoding="utf-8") as f:
            for rec in failures:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nFailures written to: {failures_path}")

    # Print report
    print()
    print(stats.summary())

    if failures:
        print(f"\n{len(failures)} failure(s) recorded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
