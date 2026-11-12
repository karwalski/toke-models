#!/usr/bin/env bash
set -euo pipefail
#
# AWS Cloud Burst Training Script for toke-model
#
# Provisions a g5.xlarge spot instance, uploads training data,
# runs QLoRA fine-tuning, and hands the results to cloud/post_train.sh, which
# uploads the adapter to the model store (MODEL_STORE_URI), pushes a pointer
# and the metrics to a results branch, and shuts the instance down.
# No laptop needed after launch.
#
# One post-train script (story 132.82): the checked-in cloud/post_train.sh is
# uploaded as-is, and cloud/run_training.sh runs training then it on the
# instance. Weights are never published as GitHub Release assets.
#
# Cost guards (132.82) — a failure must not leave a GPU instance running:
#   - here: if this script fails after it has provisioned an instance and
#     before training is launched, it terminates that instance
#     (KEEP_INSTANCE_ON_ERROR=1 keeps it, and says so);
#   - on the instance: run_training.sh arms a dead-man's switch
#     (MAX_RUNTIME_MIN, default 720) and shuts down FAILURE_GRACE_MIN
#     (default 30) minutes after any failure (KEEP_ALIVE_ON_FAILURE=1 opts
#     out). See cloud/run_training.sh.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - GitHub token that can push to GITHUB_REPO (GITHUB_TOKEN env var)
#   - MODEL_STORE_URI, and write access to it from the instance
#
# Usage:
#   ./aws_train.sh                       # Fire-and-forget: provision + train + upload + shutdown
#   ./aws_train.sh --instance-id i-xxx   # Attach to existing instance
#   ./aws_train.sh --status              # Check training status
#   ./aws_train.sh --download            # Download a pre-132.82 run's GitHub Release
#   ./aws_train.sh --terminate           # Manual terminate if needed
#
# Cost estimate: ~$3-5 total (g5.xlarge spot @ $0.35-0.50/hr x 6-10 hours)
#

# ─── Configuration ────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g5.xlarge"              # A10G 24GB, 4 vCPU, 16GB RAM
AMI_ID="${AMI_ID:-}"                    # Auto-detected if empty
KEY_NAME="${KEY_NAME:-toke-training}"   # EC2 key pair name
KEY_FILE="${KEY_FILE:-}"                # SSH key path; required, never committed (public repo)
SPOT_MAX_PRICE="0.60"                  # Max spot bid (on-demand is ~$1.01)
VOLUME_SIZE=50                         # GB — model weights (~4GB) + data + checkpoints + OS

GITHUB_REPO="${GITHUB_REPO:-karwalski/toke-models}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
MODEL_STORE_URI="${MODEL_STORE_URI:-}"
KEEP_INSTANCE_ON_ERROR="${KEEP_INSTANCE_ON_ERROR:-0}"

STATE_FILE="$(dirname "$0")/.train_state.json"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_FILE="${CORPUS_FILE:-$HOME/tk/toke-model/tokenizer/corpus.jsonl}"

REMOTE_USER="ubuntu"
REMOTE_DIR="/home/ubuntu/toke-model"
REMOTE_REPO_DIR="/home/ubuntu/toke-model-repo"   # the clone post_train.sh commits into

# ─── Helper functions ─────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ─── Local cost guard ─────────────────────────────────────────────────────────
#
# Set once this run has provisioned an instance; cleared once training is
# running on it (from then on run_training.sh's guard owns the instance).

GUARD_INSTANCE=""
ATTACHED_INSTANCE=""   # --instance-id / --skip-provision: never terminated here

on_local_exit() {
    local code=$?
    [[ $code -ne 0 ]] || return 0
    if [[ -n "$ATTACHED_INSTANCE" ]]; then
        echo "WARNING: failed with exit $code; attached instance $ATTACHED_INSTANCE is still running" \
             "and billing. Terminate it with: $0 --terminate" >&2
        return 0
    fi
    [[ -n "$GUARD_INSTANCE" ]] || return 0
    if [[ "$KEEP_INSTANCE_ON_ERROR" == 1 ]]; then
        echo "WARNING: failed with exit $code; instance $GUARD_INSTANCE is still running and billing" \
             "(KEEP_INSTANCE_ON_ERROR=1). Terminate it with: $0 --terminate" >&2
        return 0
    fi
    echo "ERROR: failed with exit $code before training started; terminating $GUARD_INSTANCE" \
         "(set KEEP_INSTANCE_ON_ERROR=1 to keep it)" >&2
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$GUARD_INSTANCE" \
        --query 'TerminatingInstances[0].CurrentState.Name' --output text >&2 \
        || echo "ERROR: terminate failed; terminate $GUARD_INSTANCE by hand" >&2
}
trap on_local_exit EXIT

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
    GUARD_INSTANCE="$INSTANCE_ID"
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

# ─── Upload the post-training scripts ────────────────────────────────────────

upload_post_train_script() {
    log "Uploading post-training scripts..."

    # The checked-in scripts, as they are: there is no second copy (132.82).
    scp_to "$SCRIPT_DIR/run_training.sh" "$REMOTE_DIR/cloud/run_training.sh"
    scp_to "$SCRIPT_DIR/post_train.sh" "$REMOTE_DIR/cloud/post_train.sh"

    # Settings for both, read on the instance by run_training.sh.
    {
        printf 'WORK_DIR=%q\n' "$REMOTE_DIR"
        printf 'REPO_DIR=%q\n' "$REMOTE_REPO_DIR"
        printf 'GITHUB_REPO=%q\n' "$GITHUB_REPO"
        printf 'MODEL_STORE_URI=%q\n' "$MODEL_STORE_URI"
        local v
        for v in MAX_RUNTIME_MIN FAILURE_GRACE_MIN KEEP_ALIVE_ON_FAILURE \
                 POST_TRAIN_PUSH POST_TRAIN_SHUTDOWN; do
            if [[ -n "${!v:-}" ]]; then printf '%s=%q\n' "$v" "${!v}"; fi
        done
    } | ssh_cmd "cat > $REMOTE_DIR/cloud/post_train.env"

    # post_train.sh pushes a results branch: give the instance a clone and
    # the token, over stdin rather than on a command line. run_training.sh
    # deletes the credential when it exits.
    printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" \
        | ssh_cmd "umask 077 && cat > ~/.git-credentials && git config --global credential.helper store"
    ssh_cmd "test -d $REMOTE_REPO_DIR/.git || git clone -q https://github.com/$GITHUB_REPO.git $REMOTE_REPO_DIR"

    log "Post-training scripts uploaded."
}

# ─── Launch fire-and-forget training ─────────────────────────────────────────

run_fire_and_forget() {
    log "Starting FIRE-AND-FORGET training on $INSTANCE_ID ($PUBLIC_IP)..."
    log "Instance will upload results to the model store and shut down when done."

    # tmux so it survives SSH disconnect. run_training.sh carries the
    # instance-side cost guard.
    ssh_cmd "bash -s" <<TRAIN_EOF
set -euo pipefail
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train bash $REMOTE_DIR/cloud/run_training.sh
TRAIN_EOF
    GUARD_INSTANCE=""      # handed over: the instance now guards itself
    ATTACHED_INSTANCE=""

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
    log "    1. Adapter weights uploaded to $MODEL_STORE_URI/<branch>/"
    log "    2. Pointer + metrics pushed to a train-results-* branch of $GITHUB_REPO"
    log "    3. Instance shuts down"
    log ""
    log "  You can close your laptop now."
    log ""
    log "  Optional monitoring (if you're curious):"
    log "    ./aws_train.sh --status"
    log "    ssh -i $KEY_FILE $REMOTE_USER@$PUBLIC_IP 'tail -f $REMOTE_DIR/train.log'"
    log ""
    log "  If anything fails, the instance shuts down after ${FAILURE_GRACE_MIN:-30} min"
    log "  (ssh in and run 'sudo shutdown -c' to keep it); it shuts down after"
    log "  ${MAX_RUNTIME_MIN:-720} min regardless. Manual cleanup: ./aws_train.sh --terminate"
    log "==========================================================="
}

# ─── Download results from GitHub ────────────────────────────────────────────
#
# Runs before 132.82 published their adapter as a GitHub Release asset; this
# fetches those. Runs since then store it at the URI in the results branch's
# adapter-weights.pointer.json.

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
tail -20 /home/ubuntu/toke-model/train.log 2>/dev/null || echo "(no log file)"
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
  Provisions GPU instance, trains, runs cloud/post_train.sh, shuts down.

Options:
  (no flags)          Fire-and-forget: provision -> train -> post_train.sh -> shutdown
  --instance-id ID    Attach to specific instance
  --skip-provision    Use existing instance from state file
  --status            Check training status
  --download          Download the latest pre-132.82 GitHub Release
  --terminate         Manually terminate instance
  --help              Show this help

Required environment:
  KEY_FILE            SSH private key path, e.g. ~/.ssh/<your-key>
                      (written there if the key pair is created by this run)
  GITHUB_TOKEN        GitHub token that can push to GITHUB_REPO (results branch)
  MODEL_STORE_URI     Where the adapter goes, e.g. s3://<bucket>/<prefix>
                      (the instance needs write access to it)

Optional environment:
  AWS_REGION          AWS region (default: us-east-1)
  AMI_ID              Override AMI (auto-detected if empty)
  KEY_NAME            EC2 key pair name (default: toke-training)
  CORPUS_FILE         Path to corpus.jsonl (default: ~/tk/toke-model/tokenizer/corpus.jsonl)
  GITHUB_REPO         GitHub repo for results branches (default: karwalski/toke-models)
  KEEP_INSTANCE_ON_ERROR=1  Keep an instance this run provisioned if launch fails
  MAX_RUNTIME_MIN     Instance dead-man's switch in minutes (default 720; 0 = off)
  FAILURE_GRACE_MIN   Minutes between a failure on the instance and shutdown (default 30)
  KEEP_ALIVE_ON_FAILURE=1   On the instance: do not shut down after a failure
  POST_TRAIN_PUSH, POST_TRAIN_SHUTDOWN   Passed to post_train.sh (see it)
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

# Validate settings before spending money on an instance.
if [[ -z "$KEY_FILE" ]]; then
    die "KEY_FILE is required: path to the SSH private key for the instance (e.g. ~/.ssh/<your-key>)."
fi
if [[ -z "$GITHUB_TOKEN" ]]; then
    die "GITHUB_TOKEN is required. Create a PAT at https://github.com/settings/tokens with 'repo' scope.\n  Export it:  export GITHUB_TOKEN=ghp_..."
fi
if [[ -z "$MODEL_STORE_URI" ]]; then
    die "MODEL_STORE_URI is required (e.g. s3://<bucket>/<prefix>): post_train.sh uploads the adapter there. Weights never go to git or a GitHub Release."
fi

# Provision or attach.
if $SKIP_PROVISION; then
    load_state || die "No state file found. Run without --skip-provision first."
    ATTACHED_INSTANCE="$INSTANCE_ID"
elif [[ -n "$INSTANCE_ID" ]]; then
    ATTACHED_INSTANCE="$INSTANCE_ID"
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
