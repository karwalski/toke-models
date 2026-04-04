# Uploading to Hugging Face Hub

Instructions for publishing toke-coder-7b to the Hugging Face Hub.

## Prerequisites

1. Install the Hugging Face CLI and git-lfs:

       pip install huggingface-hub
       brew install git-lfs
       git lfs install

2. Authenticate:

       huggingface-cli login

   You will need a write token from https://huggingface.co/settings/tokens.

## Step 1: Create the organisation (one-time)

Go to https://huggingface.co/organizations/new and create the `karwalski` organisation, or use your personal namespace.

## Step 2: Create the model repository

    huggingface-cli repo create toke-coder-7b --organization karwalski --type model

Or via the web UI at https://huggingface.co/new.

## Step 3: Clone the repo locally

    git clone https://huggingface.co/karwalski/toke-coder-7b
    cd toke-coder-7b

## Step 4: Copy model files

Copy the following into the cloned repo:

    # Model card (this is the HF README)
    cp /path/to/toke-models/huggingface/README.md .

    # Model config
    cp /path/to/toke-models/huggingface/config.json .

    # Merged model weights (from adapter merge step)
    cp -r /path/to/toke-models/merged-model/* .

    # Tokenizer files
    cp /path/to/toke-tokenizer/output/tokenizer.json .
    cp /path/to/toke-tokenizer/output/tokenizer_config.json .

## Step 5: Track large files with git-lfs

    git lfs track "*.safetensors"
    git lfs track "*.bin"
    git lfs track "*.pt"
    git lfs track "*.gguf"
    git add .gitattributes

## Step 6: Commit and push

    git add .
    git commit -m "Initial upload: toke-coder-7b (Gate 1 PASS)"
    git push

## Step 7: Verify

Visit https://huggingface.co/karwalski/toke-coder-7b and confirm:

- Model card renders correctly
- Files tab shows all weights and config
- Inference API widget loads (may take a few minutes)

## Alternative: Upload via Python API

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/path/to/merged-model",
    repo_id="karwalski/toke-coder-7b",
    repo_type="model",
)
```

## Alternative: Upload adapter only

If publishing the adapter without merged weights:

    # Copy adapter files instead of merged model
    cp -r /path/to/toke-models/results/train-results-*/adapter-mlx/* .

    # Include adapter_config.json so users can load with PEFT
    # Users will need the base model (Qwen2.5-Coder-7B) separately

## Notes

- Large files (weights) must be tracked by git-lfs. HuggingFace enforces a 10 MB limit for non-LFS files.
- The README.md in the repo root serves as the model card on HuggingFace.
- Update the model card metadata block (YAML front matter) when results change.
- Story 6.1.2 covers the actual weight and tokenizer upload.
