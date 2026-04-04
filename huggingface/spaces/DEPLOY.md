# Deploying the Toke Code Generator to HuggingFace Spaces

## Prerequisites

1. A HuggingFace account with write access to the `karwalski` namespace.
2. The HuggingFace CLI installed and authenticated:

       pip install huggingface-hub
       huggingface-cli login

3. The model `karwalski/toke-coder-7b` must already be uploaded to the Hub
   (see story 6.1.2 / `huggingface/UPLOAD.md`).

## Step 1 — Create the Space

    huggingface-cli repo create toke-code-generator \
        --organization karwalski \
        --type space \
        --space-sdk gradio

Or create it via the web UI at https://huggingface.co/new-space.

Select:
- **SDK:** Gradio
- **Hardware:** CPU basic (free tier) or T4 small for faster inference
- **Visibility:** Public

## Step 2 — Clone the Space repo

    git clone https://huggingface.co/spaces/karwalski/toke-code-generator
    cd toke-code-generator

## Step 3 — Copy files into the Space repo

    cp /path/to/toke-models/huggingface/spaces/app.py .
    cp /path/to/toke-models/huggingface/spaces/requirements.txt .
    cp /path/to/toke-models/huggingface/spaces/README.md .

## Step 4 — Commit and push

    git add .
    git commit -m "Initial deploy: toke code generator demo"
    git push

HuggingFace Spaces will automatically build and deploy the app. The build
typically takes 2-5 minutes. Monitor progress on the Space's page under the
"Logs" tab.

## Step 5 — Verify

Visit https://huggingface.co/spaces/karwalski/toke-code-generator and confirm:

- The app loads without errors.
- Entering a prompt generates toke code.
- Token stats table renders correctly.
- Example prompts work when clicked.

## Hardware considerations

The toke-coder-7b model is ~14 GB in float16. Hardware options:

| Tier | RAM | GPU | Notes |
|------|-----|-----|-------|
| CPU basic (free) | 16 GB | None | Slow inference (~30-60s). Model loaded in float32 may OOM. |
| T4 small | 16 GB | T4 16 GB | Recommended. Fast inference (~2-5s). |
| A10G small | 24 GB | A10G 24 GB | Comfortable headroom for larger generation. |

For the free tier, consider quantising the model to 4-bit (GPTQ or AWQ) and
updating `app.py` to load the quantised variant.

## Updating the Space

To push updates, modify the files locally and push:

    cd toke-code-generator
    # edit files...
    git add .
    git commit -m "Update: description of change"
    git push

The Space rebuilds automatically on every push.

## Secrets

If the model repo is private, add a `HF_TOKEN` secret in the Space settings
(Settings > Repository secrets) so the app can download model weights at
startup.

## Troubleshooting

- **Build fails:** Check the "Logs" tab for pip install errors. Ensure
  `requirements.txt` versions are compatible.
- **OOM at startup:** Switch to T4 hardware or use a quantised model.
- **Model not found:** Ensure `karwalski/toke-coder-7b` exists on the Hub and
  is accessible (public, or token secret configured).
