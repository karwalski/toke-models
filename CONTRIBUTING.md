# Contributing to toke-model

## Rules

- Never commit model weights to git. models/ is git-ignored.
- Never hardcode credentials or API keys in source files.

## Credentials and Secrets

Credentials are managed as environment variables only. Never hardcode secrets in source files or commit them to git.

Required environment variables (set these in your shell or a local `.env` file that is git-ignored):

- `ANTHROPIC_API_KEY` — Claude API key (never hardcode)
- `HF_TOKEN` — Hugging Face token for model/dataset access (never hardcode)

### Pre-commit hook setup (local secret scanning)

Install gitleaks as a pre-commit hook to catch secrets before they reach the remote:

```bash
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks
EOF
pre-commit install
```

Any secret detection finding blocks merge with zero exceptions. The CI `secret-scan` workflow enforces this on every PR.

## Testing

    python -m pytest tests/ -v

All tests must pass before any PR is merged.
