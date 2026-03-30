#!/usr/bin/env bash
# run_all_cloud.sh — Run experiments on a cloud GPU pod, push results, stop pod.
#
# Usage (from your dev box):
#   ssh -A root@HOST -p PORT -i ~/.ssh/id_ed25519 \
#     "git clone git@github.com:go37-ai/self-model.git /workspace/self-model && \
#      cd /workspace/self-model && \
#      RUNPOD_API_KEY=\$RUNPOD_API_KEY nohup bash scripts/run_all_cloud.sh \
#        --model qwen2 --experiments '1.1' > /workspace/run.log 2>&1 &"
#
# Requirements:
#   -A flag forwards your SSH agent so git push uses your GitHub SSH key
#   RUNPOD_API_KEY env var enables auto-shutdown via runpodctl
#   Clone via git@ (SSH) not https:// so push works with forwarded agent
#
# The script runs experiments, pushes results to git, then stops the pod.

set -uo pipefail

MODEL="${MODEL:-qwen2}"
EXPERIMENTS="${EXPERIMENTS:-1.1}"
PAIRS="${PAIRS:-all}"
PROFILE="cloud"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --pairs) PAIRS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Cloud experiment runner"
echo "Model: $MODEL | Profile: $PROFILE | Experiments: $EXPERIMENTS | Pairs: $PAIRS"
echo "Started: $(date -Iseconds)"
echo "============================================================"

# Source RunPod environment (provides RUNPOD_POD_ID, RUNPOD_API_KEY, etc.)
[ -f /etc/rp_environment ] && source /etc/rp_environment

# Configure runpodctl early so shutdown works even if experiments fail
if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_API_KEY:-}" ]; then
    runpodctl config --apiKey "$RUNPOD_API_KEY" 2>/dev/null || true
    echo "runpodctl configured. Pod ID: ${RUNPOD_POD_ID:-unknown}"
fi

# Ensure we're in the repo root
cd "$(dirname "$0")/.."

# Install dependencies if needed
if ! python -c "import transformers" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Track which experiments succeeded
FAILED=""

for EXP in $EXPERIMENTS; do
    echo ""
    echo "============================================================"
    echo "Running Experiment $EXP — $(date -Iseconds)"
    echo "============================================================"

    case $EXP in
        1.1)
            OUTDIR="data/results/1.1"
            if [ "$PAIRS" != "all" ]; then
                OUTDIR="data/results/1.1_${PAIRS}"
            fi
            python scripts/01_extract_vector.py --profile "$PROFILE" --model "$MODEL" --pairs "$PAIRS" \
                --output-dir "$OUTDIR" \
                || FAILED="$FAILED 1.1"
            ;;
        1.2)
            python scripts/02_pca_persona_space.py --profile "$PROFILE" --model "$MODEL" \
                || FAILED="$FAILED 1.2"
            ;;
        1.3)
            python scripts/03_blackmail_validation.py --profile "$PROFILE" --model "$MODEL" \
                || FAILED="$FAILED 1.3"
            ;;
        *)
            echo "Unknown experiment: $EXP (skipping)"
            ;;
    esac
done

echo ""
echo "============================================================"
echo "Experiments complete — $(date -Iseconds)"
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
fi
echo "============================================================"

# Upload results to S3 (all files including .pt vectors)
S3_BUCKET="${S3_RESULTS_BUCKET:-go37-ai}"
S3_PREFIX="self-model-results"
UPLOAD_OK=false

if command -v aws &>/dev/null && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "Uploading results to s3://${S3_BUCKET}/${S3_PREFIX}/..."
    # Sync all result dirs including activations (small enough with layer_stride)
    aws s3 sync data/results/ "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        && UPLOAD_OK=true \
        || echo "WARNING: S3 upload failed."
else
    echo "AWS CLI not configured. Installing..."
    pip install awscli 2>/dev/null
    if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
        aws s3 sync data/results/ "s3://${S3_BUCKET}/${S3_PREFIX}/" \
            && UPLOAD_OK=true \
            || echo "WARNING: S3 upload failed."
    else
        echo "WARNING: AWS_ACCESS_KEY_ID not set. Cannot upload to S3."
    fi
fi

# Also push small files (JSON, logs) to git as backup
echo "Pushing small results to git..."
git config user.email "cloud-runner@self-model"
git config user.name "Cloud Runner"
git add data/results/ -f
git reset -- 'data/results/*/activations/' 2>/dev/null || true
git reset -- 'data/results/**/*.pt' 2>/dev/null || true
git add -u

if git diff --cached --quiet; then
    echo "No new results to push to git."
else
    git commit -m "Cloud run results: experiments ${EXPERIMENTS} on ${MODEL} ($(date -I))"
    git push origin main && echo "Git push succeeded." || echo "WARNING: git push failed (results are in S3)."
fi

# Only shut down if results were exported
if [ "$UPLOAD_OK" != "true" ]; then
    echo "ERROR: Results not uploaded. NOT shutting down pod."
    echo "Download results manually, then stop the pod from the dashboard."
    exit 1
fi

# Stop THIS pod via runpodctl
# RUNPOD_POD_ID is injected by RunPod but only available in interactive shells.
# Try sourcing common profile locations to pick it up.
echo "Stopping pod..."
if [ -z "${RUNPOD_POD_ID:-}" ]; then
    for f in /etc/rp_environment /etc/environment /root/.bashrc; do
        [ -f "$f" ] && source "$f" 2>/dev/null
    done
fi

if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod $RUNPOD_POD_ID..."
    runpodctl stop pod "$RUNPOD_POD_ID"
elif command -v runpodctl &>/dev/null; then
    # Fallback: if only one RUNNING pod, stop it
    RUNNING=$(runpodctl get pod 2>/dev/null | awk '$NF=="RUNNING" {print $1}')
    NUM=$(echo "$RUNNING" | grep -c . || true)
    if [ "$NUM" = "1" ]; then
        echo "One running pod found: $RUNNING. Stopping..."
        runpodctl stop pod "$RUNNING"
    else
        echo "Cannot identify this pod. RUNPOD_POD_ID not set, $NUM running pods found."
        echo "Stop the pod manually."
    fi
else
    echo "runpodctl not available. Stop the pod manually."
fi
