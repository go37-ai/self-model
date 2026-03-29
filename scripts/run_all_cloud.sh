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

# Configure runpodctl early so shutdown works even if experiments fail
if command -v runpodctl &>/dev/null && [ -n "${RUNPOD_API_KEY:-}" ]; then
    runpodctl config --apiKey "$RUNPOD_API_KEY" 2>/dev/null || true
    echo "runpodctl configured."
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

# Push results to git (small files only — skip activations cache)
echo "Pushing results to git..."
git config user.email "cloud-runner@self-model"
git config user.name "Cloud Runner"

# Note: push auth comes from the clone URL. Clone with token:
#   git clone https://TOKEN@github.com/go37-ai/self-model.git

git add data/results/ -f
git reset -- 'data/results/*/activations/' 2>/dev/null || true
git add -u

if git diff --cached --quiet; then
    echo "No new results to push."
else
    git commit -m "Cloud run results: experiments ${EXPERIMENTS} on ${MODEL} ($(date -I))"
    if git push origin main; then
        echo "Results pushed successfully."
        PUSH_OK=true
    else
        echo "ERROR: git push failed. Results are on disk. NOT shutting down pod."
        echo "Download results manually, then stop the pod from the dashboard."
        exit 1
    fi
fi

# Stop THIS pod via runpodctl (only if push succeeded or no results to push)
# Match pod by hostname to avoid stopping other running pods
echo "Stopping pod..."
if command -v runpodctl &>/dev/null; then
    HOSTNAME=$(hostname)
    # runpodctl get pod output: ID, NAME, GPU, IMAGE, STATUS
    # Pod IDs contain the hostname prefix in container environments
    POD_ID=$(runpodctl get pod 2>/dev/null | awk -v host="$HOSTNAME" 'NR>1 {print $1}' | while read id; do
        if echo "$HOSTNAME" | grep -q "${id:0:12}" 2>/dev/null || echo "$id" | grep -q "${HOSTNAME:0:12}" 2>/dev/null; then
            echo "$id"
            break
        fi
    done)
    # Fallback: if only one pod is running, stop it
    if [ -z "$POD_ID" ]; then
        NUM_PODS=$(runpodctl get pod 2>/dev/null | awk 'NR>1' | wc -l)
        if [ "$NUM_PODS" = "1" ]; then
            POD_ID=$(runpodctl get pod 2>/dev/null | awk 'NR==2 {print $1}')
            echo "Only one pod running, assuming it's this one."
        else
            echo "Multiple pods running and cannot identify this one. Stop manually."
            echo "Hostname: $HOSTNAME"
            runpodctl get pod 2>/dev/null
            exit 0
        fi
    fi
    if [ -n "$POD_ID" ]; then
        echo "Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID"
    fi
else
    echo "runpodctl not available. Stop the pod manually."
fi
