#!/usr/bin/env bash
set -euo pipefail
#
# AWS Cloud Burst Training Script for toke-models
#
# Provisions a g5.xlarge spot instance, uploads training data,
# runs QLoRA fine-tuning, uploads results to GitHub Release,
# and self-terminates. No laptop needed after launch.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - GitHub token with repo + write:packages scope (GITHUB_TOKEN env var)
#
# Usage:
#   ./aws_train.sh                       # Fire-and-forget: provision + train + upload + shutdown
#   ./aws_train.sh --instance-id i-xxx   # Attach to existing instance
#   ./aws_train.sh --status              # Check training status
#   ./aws_train.sh --download            # Download results locally from GitHub release
#   ./aws_train.sh --terminate           # Manual terminate if needed
#
# Cost estimate: ~$3-5 total (g5.xlarge spot @ $0.35-0.50/hr x 6-10 hours)
#

# ─── Configuration ────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g5.xlarge"              # A10G 24GB, 4 vCPU, 16GB RAM
AMI_ID="${AMI_ID:-}"                    # Auto-detected if empty
KEY_NAME="${KEY_NAME:-toke-training}"   # EC2 key pair name
KEY_FILE="${KEY_FILE:-$HOME/.ssh/toke-training.pem}"
SPOT_MAX_PRICE="0.60"                  # Max spot bid (on-demand is ~$1.01)
VOLUME_SIZE=50                         # GB — model weights (~4GB) + data + checkpoints + OS

GITHUB_REPO="${GITHUB_REPO:-karwalski/toke-models}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

STATE_FILE="$(dirname "$0")/.train_state.json"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_FILE="${CORPUS_FILE:-$HOME/tk/toke-tokenizer/corpus.jsonl}"

REMOTE_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/toke-models"

# ─── Helper functions ─────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

save_state() {
    cat > "$STATE_FILE" <<STEOF
{
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP",
  "spot_request_id": "${SPOT_REQUEST_ID:-}",
  "region": "$REGION",
  "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
STEOF
}

load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        INSTANCE_ID=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('instance_id',''))")
        PUBLIC_IP=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('public_ip',''))")
        SPOT_REQUEST_ID=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('spot_request_id',''))")
        return 0
    fi
    return 1
}

wait_for_ssh() {
    local ip=$1
    local max_wait=300
    local elapsed=0
    log "Waiting for SSH on $ip..."
    while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$KEY_FILE" "$REMOTE_USER@$ip" "echo ok" &>/dev/null; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [[ $elapsed -ge $max_wait ]]; then
            die "SSH timeout after ${max_wait}s"
        fi
    done
    log "SSH ready."
}

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "$REMOTE_USER@$PUBLIC_IP" "$@"
}

scp_to() {
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" "$1" "$REMOTE_USER@$PUBLIC_IP:$2"
}

scp_from() {
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" -r "$REMOTE_USER@$PUBLIC_IP:$1" "$2"
}

# ─── Auto-detect AMI ──────────────────────────────────────────────────────────

detect_ami() {
    if [[ -n "$AMI_ID" ]]; then
        return
    fi
    log "Auto-detecting Deep Learning AMI in $REGION..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=x86_64" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text 2>/dev/null || true)

    if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
        AMI_ID=$(aws ec2 describe-images \
            --region "$REGION" \
            --owners 099720109477 \
            --filters \
                "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
                "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text)
        log "Using Ubuntu 22.04 AMI: $AMI_ID (will install CUDA manually)"
    else
        log "Using Deep Learning AMI: $AMI_ID"
    fi
}

# ─── Create key pair if needed ────────────────────────────────────────────────

ensure_key_pair() {
    if [[ -f "$KEY_FILE" ]]; then
        return
    fi
    log "Creating EC2 key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    log "Key saved to $KEY_FILE"
}

# ─── Create security group ───────────────────────────────────────────────────

ensure_security_group() {
    SG_NAME="toke-training-sg"
    SG_ID=$(aws ec2 describe-security-groups \
        --region "$REGION" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")

    if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
        log "Creating security group: $SG_NAME"
        SG_ID=$(aws ec2 create-security-group \
            --region "$REGION" \
            --group-name "$SG_NAME" \
            --description "toke model training - SSH only" \
            --query 'GroupId' \
            --output text)
        aws ec2 authorize-security-group-ingress \
            --region "$REGION" \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0
    fi
    log "Security group: $SG_ID"
}

# ─── Provision spot instance ──────────────────────────────────────────────────

provision_spot() {
    detect_ami
    ensure_key_pair
    ensure_security_group

    log "Requesting g5.xlarge spot instance (max bid: \$$SPOT_MAX_PRICE/hr)..."

    SPOT_REQUEST_ID=$(aws ec2 request-spot-instances \
        --region "$REGION" \
        --spot-price "$SPOT_MAX_PRICE" \
        --instance-count 1 \
        --type "one-time" \
        --launch-specification "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"KeyName\": \"$KEY_NAME\",
            \"SecurityGroupIds\": [\"$SG_ID\"],
            \"BlockDeviceMappings\": [{
                \"DeviceName\": \"/dev/sda1\",
                \"Ebs\": {
                    \"VolumeSize\": $VOLUME_SIZE,
                    \"VolumeType\": \"gp3\",
                    \"DeleteOnTermination\": true
                }
            }]
        }" \
        --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
        --output text)

    log "Spot request: $SPOT_REQUEST_ID"
    log "Waiting for fulfillment..."

    aws ec2 wait spot-instance-request-fulfilled \
        --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST_ID" \
        --cli-read-timeout 300 2>/dev/null || true

    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST_ID" \
        --query 'SpotInstanceRequests[0].InstanceId' \
        --output text)

    if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
        die "Spot request not fulfilled. Check AWS console for capacity issues."
    fi

    log "Instance: $INSTANCE_ID"
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "Public IP: $PUBLIC_IP"
    save_state

    aws ec2 create-tags \
        --region "$REGION" \
        --resources "$INSTANCE_ID" \
        --tags Key=Name,Value=toke-training Key=Project,Value=toke
}

# ─── Setup remote environment ────────────────────────────────────────────────

setup_remote() {
    log "Setting up remote environment..."

    ssh_cmd "bash -s" <<'SETUP_EOF'
set -euo pipefail

# Check GPU.
if ! nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Installing CUDA drivers..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-driver-535 nvidia-cuda-toolkit
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create virtualenv.
python3 -m venv ~/toke-venv
source ~/toke-venv/bin/activate

# Install training dependencies.
pip install -q --upgrade pip
pip install -q \
    torch \
    transformers \
    datasets \
    peft \
    bitsandbytes \
    accelerate \
    sentencepiece \
    pyyaml \
    scipy

echo "Remote setup complete."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
SETUP_EOF
}

# ─── Upload training data and scripts ────────────────────────────────────────

upload_data() {
    log "Preparing training data locally..."

    local train_data="$PROJECT_DIR/training-data/train.jsonl"
    if [[ ! -f "$train_data" ]]; then
        log "Generating training data from corpus..."
        cd "$PROJECT_DIR"
        python3 finetune/prepare_data.py \
            --corpus "$CORPUS_FILE" \
            --output-dir training-data/
    fi

    log "Uploading files to instance..."
    ssh_cmd "mkdir -p $REMOTE_DIR/{finetune/configs,training-data,output,cloud}"

    scp_to "$PROJECT_DIR/finetune/train_qlora.py" "$REMOTE_DIR/finetune/"
    scp_to "$PROJECT_DIR/finetune/merge_adapters.py" "$REMOTE_DIR/finetune/"
    scp_to "$PROJECT_DIR/finetune/configs/7b.yaml" "$REMOTE_DIR/finetune/configs/"
    scp_to "$PROJECT_DIR/training-data/train.jsonl" "$REMOTE_DIR/training-data/"
    scp_to "$PROJECT_DIR/training-data/eval.jsonl" "$REMOTE_DIR/training-data/"

    log "Upload complete."
}

# ─── Upload the post-training script ─────────────────────────────────────────

upload_post_train_script() {
    log "Uploading post-training script..."

    # Write the script that runs on the instance after training completes.
    # It packages results, uploads to GitHub Release, and shuts down.
    cat <<'POST_SCRIPT_EOF' | ssh_cmd "cat > $REMOTE_DIR/cloud/post_train.sh && chmod +x $REMOTE_DIR/cloud/post_train.sh"
#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/home/ubuntu/toke-models"
GITHUB_TOKEN_FILE="$WORK_DIR/.github_token"
GITHUB_REPO_FILE="$WORK_DIR/.github_repo"

log() { echo "[$(date '+%H:%M:%S')] POST-TRAIN: $*" | tee -a "$WORK_DIR/train.log"; }

GITHUB_TOKEN=$(cat "$GITHUB_TOKEN_FILE" 2>/dev/null || echo "")
GITHUB_REPO=$(cat "$GITHUB_REPO_FILE" 2>/dev/null || echo "")

if [[ -z "$GITHUB_TOKEN" || -z "$GITHUB_REPO" ]]; then
    log "ERROR: Missing GitHub token or repo. Skipping upload. Instance will NOT shut down."
    exit 1
fi

RELEASE_TAG="train-$(date -u +%Y%m%d-%H%M%S)"
ADAPTER_DIR="$WORK_DIR/output/7b-qlora/adapter"
PACKAGE_DIR="$WORK_DIR/output/release-package"

log "Packaging results for GitHub release $RELEASE_TAG..."

mkdir -p "$PACKAGE_DIR"

# Collect training metrics.
cp "$WORK_DIR/output/7b-qlora/training_metrics.json" "$PACKAGE_DIR/" 2>/dev/null || true
cp "$WORK_DIR/output/7b-qlora/eval_metrics.json" "$PACKAGE_DIR/" 2>/dev/null || true
cp "$WORK_DIR/train.log" "$PACKAGE_DIR/" 2>/dev/null || true

# Tar the adapter weights (typically 100-300MB).
if [[ -d "$ADAPTER_DIR" ]]; then
    log "Compressing adapter weights..."
    cd "$WORK_DIR/output/7b-qlora"
    tar czf "$PACKAGE_DIR/adapter-weights.tar.gz" adapter/
    ADAPTER_SIZE=$(du -sh "$PACKAGE_DIR/adapter-weights.tar.gz" | cut -f1)
    log "Adapter archive: $ADAPTER_SIZE"
else
    log "WARNING: No adapter directory found at $ADAPTER_DIR"
fi

# Build release notes from metrics.
NOTES="## QLoRA Training Results\n\n"
NOTES+="- **Model:** Qwen 2.5 Coder 7B Instruct\n"
NOTES+="- **Method:** QLoRA (4-bit NF4, rank 64)\n"
NOTES+="- **Corpus:** 46,754 toke programs\n"
NOTES+="- **Training examples:** ~73K (direct + comparison)\n"
NOTES+="- **Instance:** $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'g5.xlarge')\n"
NOTES+="- **GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')\n\n"

if [[ -f "$PACKAGE_DIR/training_metrics.json" ]]; then
    TRAIN_LOSS=$(python3 -c "import json; d=json.load(open('$PACKAGE_DIR/training_metrics.json')); print(f\"{d.get('train_loss', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
    NOTES+="- **Final train loss:** $TRAIN_LOSS\n"
fi
if [[ -f "$PACKAGE_DIR/eval_metrics.json" ]]; then
    EVAL_LOSS=$(python3 -c "import json; d=json.load(open('$PACKAGE_DIR/eval_metrics.json')); print(f\"{d.get('eval_loss', 'N/A'):.4f}\")" 2>/dev/null || echo "N/A")
    NOTES+="- **Eval loss:** $EVAL_LOSS\n"
fi

NOTES+="\n### Assets\n"
NOTES+="- \`adapter-weights.tar.gz\` — LoRA adapter weights (merge with base model using merge_adapters.py)\n"
NOTES+="- \`training_metrics.json\` — Training loss curve\n"
NOTES+="- \`eval_metrics.json\` — Evaluation metrics\n"
NOTES+="- \`train.log\` — Full training log\n"

# Install gh CLI if not present.
if ! command -v gh &>/dev/null; then
    log "Installing GitHub CLI..."
    (type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y))
    sudo mkdir -p -m 755 /etc/apt/keyrings
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
    sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
    sudo apt update -qq && sudo apt install gh -y -qq
fi

# Authenticate gh.
echo "$GITHUB_TOKEN" | gh auth login --with-token

# Create GitHub release with assets.
log "Creating GitHub release $RELEASE_TAG on $GITHUB_REPO..."
RELEASE_NOTES=$(echo -e "$NOTES")

cd "$PACKAGE_DIR"
ASSET_FLAGS=""
for f in adapter-weights.tar.gz training_metrics.json eval_metrics.json train.log; do
    if [[ -f "$f" ]]; then
        ASSET_FLAGS="$ASSET_FLAGS $f"
    fi
done

gh release create "$RELEASE_TAG" \
    --repo "$GITHUB_REPO" \
    --title "Training run $RELEASE_TAG" \
    --notes "$RELEASE_NOTES" \
    $ASSET_FLAGS

log "Release created: https://github.com/$GITHUB_REPO/releases/tag/$RELEASE_TAG"

# Clean up token.
rm -f "$GITHUB_TOKEN_FILE"

log "Upload complete. Shutting down instance in 60 seconds..."
log "(Cancel with: sudo shutdown -c)"
sudo shutdown -h +1 "Training complete. Results uploaded to GitHub. Shutting down."
POST_SCRIPT_EOF

    # Upload GitHub credentials securely (deleted after use by post_train.sh).
    ssh_cmd "echo '$GITHUB_TOKEN' > $REMOTE_DIR/.github_token && chmod 600 $REMOTE_DIR/.github_token"
    ssh_cmd "echo '$GITHUB_REPO' > $REMOTE_DIR/.github_repo"

    log "Post-training script uploaded."
}

# ─── Launch fire-and-forget training ─────────────────────────────────────────

run_fire_and_forget() {
    log "Starting FIRE-AND-FORGET training on $INSTANCE_ID ($PUBLIC_IP)..."
    log "Instance will upload results to GitHub and self-terminate when done."

    ssh_cmd "bash -s" <<TRAIN_EOF
set -euo pipefail

# Run everything in tmux so it survives SSH disconnect.
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train bash -c '
    set -euo pipefail
    source ~/toke-venv/bin/activate
    cd $REMOTE_DIR

    echo "=== Training started at \$(date -u) ===" | tee train.log
    echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" | tee -a train.log

    # Run training.
    python3 finetune/train_qlora.py --config finetune/configs/7b.yaml 2>&1 | tee -a train.log
    TRAIN_EXIT=\$?

    echo "" | tee -a train.log
    echo "=== Training finished at \$(date -u) with exit code \$TRAIN_EXIT ===" | tee -a train.log

    if [[ \$TRAIN_EXIT -eq 0 ]]; then
        echo "=== Starting post-training upload ===" | tee -a train.log
        bash cloud/post_train.sh 2>&1 | tee -a train.log
    else
        echo "=== Training FAILED. Instance will NOT shut down. ===" | tee -a train.log
        echo "SSH in to investigate: ssh -i KEY ubuntu@$PUBLIC_IP" | tee -a train.log
    fi
'
TRAIN_EOF

    log ""
    log "==========================================================="
    log "  FIRE-AND-FORGET TRAINING LAUNCHED"
    log "==========================================================="
    log ""
    log "  Instance:  $INSTANCE_ID ($PUBLIC_IP)"
    log "  GPU:       A10G 24GB"
    log "  ETA:       6-10 hours"
    log "  Cost:      ~\$3-5"
    log ""
    log "  When training completes:"
    log "    1. Adapter weights + metrics uploaded to GitHub Release"
    log "       https://github.com/$GITHUB_REPO/releases"
    log "    2. Instance self-terminates"
    log ""
    log "  You can close your laptop now."
    log ""
    log "  Optional monitoring (if you're curious):"
    log "    ./aws_train.sh --status"
    log "    ssh -i $KEY_FILE $REMOTE_USER@$PUBLIC_IP 'tail -f $REMOTE_DIR/train.log'"
    log ""
    log "  If training fails, instance stays running for debugging."
    log "  Manual cleanup: ./aws_train.sh --terminate"
    log "==========================================================="
}

# ─── Download results from GitHub ────────────────────────────────────────────

download_from_github() {
    if [[ -z "$GITHUB_TOKEN" ]]; then
        die "GITHUB_TOKEN required. Set it in your environment."
    fi

    log "Fetching latest training release from $GITHUB_REPO..."

    local latest_tag
    latest_tag=$(gh release list --repo "$GITHUB_REPO" --limit 1 --json tagName -q '.[0].tagName' 2>/dev/null || echo "")

    if [[ -z "$latest_tag" ]]; then
        die "No releases found on $GITHUB_REPO"
    fi

    log "Latest release: $latest_tag"

    local local_output="$PROJECT_DIR/output/$latest_tag"
    mkdir -p "$local_output"

    gh release download "$latest_tag" \
        --repo "$GITHUB_REPO" \
        --dir "$local_output"

    # Extract adapter weights if present.
    if [[ -f "$local_output/adapter-weights.tar.gz" ]]; then
        log "Extracting adapter weights..."
        cd "$local_output"
        tar xzf adapter-weights.tar.gz
        log "Adapter extracted to $local_output/adapter/"
    fi

    log "Results downloaded to $local_output/"
}

# ─── Terminate instance ──────────────────────────────────────────────────────

terminate() {
    if [[ -z "${INSTANCE_ID:-}" ]]; then
        load_state || die "No instance state found"
    fi

    log "Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'TerminatingInstances[0].CurrentState.Name' \
        --output text

    if [[ -n "${SPOT_REQUEST_ID:-}" ]]; then
        aws ec2 cancel-spot-instance-requests \
            --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" &>/dev/null || true
    fi

    rm -f "$STATE_FILE"
    log "Instance terminated. Spot request cancelled."
}

# ─── Check training status ───────────────────────────────────────────────────

check_status() {
    load_state || die "No instance state found"

    # First check if instance is still running.
    local state
    state=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "unknown")

    if [[ "$state" != "running" ]]; then
        log "Instance $INSTANCE_ID is $state (likely self-terminated after uploading results)"
        log "Download results: $0 --download"
        return 0
    fi

    log "Checking training status on $PUBLIC_IP..."
    ssh_cmd "bash -s" <<'STATUS_EOF'
echo ""
if tmux has-session -t train 2>/dev/null; then
    echo "STATUS: TRAINING IN PROGRESS (tmux session active)"
else
    echo "STATUS: tmux session ended (training complete or failed)"
fi
echo ""
echo "=== Last 20 lines of train.log ==="
tail -20 /home/ubuntu/toke-models/train.log 2>/dev/null || echo "(no log file)"
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
STATUS_EOF
}

# ─── Main ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

  Fire-and-forget cloud training for toke model fine-tuning.
  Provisions GPU instance, trains, uploads results to GitHub, self-terminates.

Options:
  (no flags)          Fire-and-forget: provision -> train -> upload to GitHub -> shutdown
  --instance-id ID    Attach to specific instance
  --skip-provision    Use existing instance from state file
  --status            Check training status
  --download          Download results from latest GitHub release
  --terminate         Manually terminate instance
  --help              Show this help

Required environment:
  GITHUB_TOKEN        GitHub PAT with repo scope (for uploading results)

Optional environment:
  AWS_REGION          AWS region (default: us-east-1)
  AMI_ID              Override AMI (auto-detected if empty)
  KEY_NAME            EC2 key pair name (default: toke-training)
  KEY_FILE            SSH key file path (default: ~/.ssh/toke-training.pem)
  CORPUS_FILE         Path to corpus.jsonl (default: ~/tk/toke-tokenizer/corpus.jsonl)
  GITHUB_REPO         GitHub repo for releases (default: karwalski/toke-models)
EOF
}

SKIP_PROVISION=false
DO_DOWNLOAD=false
DO_STATUS=false
DO_TERMINATE=false
INSTANCE_ID=""
PUBLIC_IP=""
SPOT_REQUEST_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-provision) SKIP_PROVISION=true; shift ;;
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --download) DO_DOWNLOAD=true; shift ;;
        --status) DO_STATUS=true; shift ;;
        --terminate) DO_TERMINATE=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# Single-action commands.
if $DO_STATUS; then
    check_status
    exit 0
fi

if $DO_TERMINATE; then
    terminate
    exit 0
fi

if $DO_DOWNLOAD; then
    download_from_github
    exit 0
fi

# Validate GitHub token before spending money on an instance.
if [[ -z "$GITHUB_TOKEN" ]]; then
    die "GITHUB_TOKEN is required. Create a PAT at https://github.com/settings/tokens with 'repo' scope.\n  Export it:  export GITHUB_TOKEN=ghp_..."
fi

# Provision or attach.
if $SKIP_PROVISION; then
    load_state || die "No state file found. Run without --skip-provision first."
elif [[ -n "$INSTANCE_ID" ]]; then
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    save_state
else
    provision_spot
fi

wait_for_ssh "$PUBLIC_IP"
setup_remote
upload_data
upload_post_train_script
run_fire_and_forget

log "Done. You can close your laptop."
