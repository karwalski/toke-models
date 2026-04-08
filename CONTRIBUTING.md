# Contributing to toke-model

## Rules

- Model weights, adapters, and training data are never committed to git.
- All Python code must pass `ruff check .` and `mypy .` before commit.
- Evaluation results: commit only aggregated reports, not raw output files.
- Changes to training configs require a documented rationale.

## Testing

    python -m pytest tests/ -v

## Developer Certificate of Origin

Sign your commits: `git commit -s`
