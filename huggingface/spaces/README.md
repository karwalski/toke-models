---
title: Toke Code Generator
emoji: "\U0001F4BB"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - karwalski/toke-coder-7b
short_description: Generate toke source code from natural language descriptions
---

# Toke Code Generator

Generate **toke** programming language code from natural language task descriptions, powered by [toke-coder-7b](https://huggingface.co/karwalski/toke-coder-7b).

## What is toke?

Toke is a research programming language designed for concise, structured program representation with measurable token efficiency gains over general-purpose languages. The toke-coder-7b model is a QLoRA/DoRA fine-tune of Qwen 2.5 Coder 7B trained on 46,000+ compiler-validated toke programs.

## Features

- Enter a natural language task description and receive generated toke code.
- Syntax-highlighted code output.
- Token count comparison showing toke vs estimated Python token usage.
- Pre-filled example prompts for quick exploration.

## Links

- Model: [karwalski/toke-coder-7b](https://huggingface.co/karwalski/toke-coder-7b)
- Language: [github.com/karwalski](https://github.com/karwalski)
