"""
Toke Code Generator — HuggingFace Spaces Demo

Generates toke programming language code from natural language descriptions
using the fine-tuned toke-coder-7b model.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_ID = "karwalski/toke-coder-7b"

_model = None
_tokenizer = None


def _load_model():
    """Load model and tokenizer once, caching for subsequent calls."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _count_toke_tokens(code: str, tokenizer) -> int:
    """Count tokens in the generated toke code."""
    return len(tokenizer.encode(code, add_special_tokens=False))


# Removed 2026-09-19 (story 132.15): `_estimate_python_tokens()` synthesised a Python
# token count by dividing the toke count by 0.875 — the withdrawn Gate 1 "12.5%
# reduction" — and the page then displayed the arithmetic back as a measured "token
# reduction". It was a claim with no measurement behind it, and its direction is wrong:
# under one shared tokenizer (cl100k_base) toke costs 1.34x [1.22, 1.48] the tokens of
# equivalent Python on the 60 Gate-1 tasks (N = 60, 2026-09-19). A real comparison needs
# a real Python program tokenized by the same tokenizer; see
# toke/docs/metrics-baseline.md.


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_toke(prompt: str, max_tokens: int, temperature: float) -> tuple:
    """Generate toke code from a natural language description.

    Returns:
        tuple of (generated_code, stats_markdown)
    """
    if not prompt.strip():
        return "", ""

    model, tokenizer = _load_model()

    full_prompt = (
        "Write toke code for the following task.\n\n"
        f"Task: {prompt.strip()}\n\n"
        "```toke\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Trim at closing fence if the model emits one.
    if "```" in generated:
        generated = generated[:generated.index("```")]

    generated = generated.rstrip()

    # --- token stats ---
    toke_count = _count_toke_tokens(generated, tokenizer)

    stats = (
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Toke tokens (this model's tokenizer) | {toke_count} |\n"
    )

    return generated, stats


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["A function that returns the factorial of n"],
    ["A struct representing a 2-D point with an add method"],
    ["FizzBuzz from 1 to 100"],
    ["Read lines from a file and return the longest line"],
    ["A generic stack with push, pop, and is_empty"],
    ["Binary search over a sorted list of integers"],
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Toke Code Generator",
    theme=gr.themes.Monochrome(),
) as demo:
    gr.Markdown(
        "# Toke Code Generator\n"
        "Generate **toke** source code from a natural language description.  \n"
        "Powered by [toke-coder-7b](https://huggingface.co/karwalski/toke-coder-7b) "
        "— a QLoRA/DoRA fine-tune of Qwen 2.5 Coder 7B trained on 46 000+ "
        "validated toke programs."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Task description",
                placeholder="Describe what the toke program should do...",
                lines=3,
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=256,
                    step=64,
                    label="Max new tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.4,
                    step=0.1,
                    label="Temperature",
                )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_code = gr.Code(
                label="Generated toke code",
                language=None,
                lines=16,
            )
            stats_box = gr.Markdown(label="Token stats")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt],
        label="Example prompts",
    )

    generate_btn.click(
        fn=generate_toke,
        inputs=[prompt, max_tokens, temperature],
        outputs=[output_code, stats_box],
    )
    prompt.submit(
        fn=generate_toke,
        inputs=[prompt, max_tokens, temperature],
        outputs=[output_code, stats_box],
    )

    gr.Markdown(
        "---\n"
        "*toke* is a research programming language focused on token efficiency.  \n"
        "Learn more: [github.com/karwalski](https://github.com/karwalski) "
        "| Model card: [karwalski/toke-coder-7b](https://huggingface.co/karwalski/toke-coder-7b)"
    )

if __name__ == "__main__":
    demo.launch()
