# Contributing to toke-tokenizer

## Rules

- Tokenizer vocabulary files (.model, .vocab) are never committed to git.
- All Python code must pass `ruff check .` and `mypy .` before commit.
- Changes to tokenizer training parameters require a documented rationale
  in docs/tokenizer-design.md.

## Testing

    python -m pytest tests/ -v

## Developer Certificate of Origin

Sign your commits: `git commit -s`
