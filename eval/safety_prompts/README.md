# Adversarial Safety Prompt Templates

This directory contains adversarial prompt templates used in toke model safety evaluation.
See `docs/security/model-safety-evals.md` for the full evaluation process.

---

## Template Format

Each template is a JSON object in `harmful_code_templates.json` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | One of the five category names (see below) |
| `template` | string | The prompt with a `{TASK}` placeholder |
| `expected_harmful` | boolean | Always `true` for adversarial templates |

The `{TASK}` placeholder is replaced at evaluation time with task-specific variants to
expand the 50 base templates to the full evaluation corpus (500+ prompts per category).
The placeholder allows one template to cover many surface variations without duplicating
the underlying intent.

---

## The Five Adversarial Categories

### `exploit_scaffolding`

Prompts that ask for toke programs implementing components of binary exploitation:
buffer overflow setups, ROP-chain-style control flow manipulation, shellcode stubs,
or integer overflow conditions designed to produce controlled crashes or arbitrary
code execution.

### `malware_structures`

Prompts that ask for toke programs implementing core malware behaviors: keylogging
(capturing input streams), file encryption with key-controlled decryption (ransomware
pattern), persistence via startup-path writes, or dropper logic that downloads and
executes external code.

### `data_exfiltration`

Prompts that ask for toke programs that read sensitive files or environment variables
and transmit their contents over the network, embed them in output, or write them to
world-readable locations.

### `sandbox_escape`

Prompts that ask for toke programs designed to detect or defeat sandboxing: probing
for sandbox-exec restrictions, attempting Docker container escape, privilege escalation,
or host interaction via side channels (timing, resource exhaustion).

### `credential_harvesting`

Prompts that ask for toke programs targeting specific developer-environment secrets:
Anthropic API keys, HuggingFace tokens, SSH private keys, `.env` file contents, or
cloud provider credentials. Distinguished from general exfiltration by targeting
specific known secret patterns.

---

## Adding New Templates

New templates must:
1. Belong to one of the five existing categories
2. Include a `{TASK}` placeholder that varies the surface form without changing the
   underlying harmful intent
3. Have `"expected_harmful": true`
4. Be abstract enough that the template text itself does not constitute harmful instructions

Do not add templates that describe specific CVEs, working exploit techniques, or real
malware families in implementation detail. Templates should be phrased as task descriptions,
not as implementation guides.

---

## Expanding to Full Corpus

The `eval/safety_eval.py` script expands templates to corpus scale using a built-in
variant generator. For each template, it generates `prompts_per_category / n_templates`
variants by substituting different `{TASK}` values drawn from a task-variation list
appropriate to each category.

In dry-run mode (`--dry-run`), only 10 prompts per category (using the base templates
directly) are generated, and both model inference and LlamaGuard classification are
stubbed.
