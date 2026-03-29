#!/usr/bin/env bash
# run_all_cloud.sh — Run experiments on a cloud GPU pod, push results, stop pod.
#
# Usage:
#   bash scripts/run_all_cloud.sh [--model MODEL] [--experiments EXPERIMENTS]
#
# Examples:
#   bash scripts/run_all_cloud.sh                          # Run all experiments, qwen2
#   bash scripts/run_all_cloud.sh --model qwen2 --experiments "1.1"
#   bash scripts/run_all_cloud.sh --model qwen3 --experiments "1.1 1.2 1.3"
#
# This script is designed for disposable pods: clone repo, install deps,
# run experiments, push results to git, stop the pod.

set -euo pipefail

MODEL="${MODEL:-qwen2}"
EXPERIMENTS="${EXPERIMENTS:-1.1}"
PROFILE="cloud"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Cloud experiment runner"
echo "Model: $MODEL | Profile: $PROFILE | Experiments: $EXPERIMENTS"
echo "Started: $(date -Iseconds)"
echo "============================================================"

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
            python scripts/01_extract_vector.py --profile "$PROFILE" --model "$MODEL" \
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

# Push results to git (small files only — vectors, JSON, logs)
echo "Pushing results to git..."
git config user.email "cloud-runner@self-model"
git config user.name "Cloud Runner"
git add data/results/ -f
git add -u  # pick up any modified tracked files

if git diff --cached --quiet; then
    echo "No new results to push."
else
    git commit -m "Cloud run results: experiments ${EXPERIMENTS} on ${MODEL} ($(date -I))"
    git push origin main
    echo "Results pushed."
fi

# Stop the pod via RunPod API
# Requires RUNPOD_API_KEY env var (set in pod template or passed at launch)
# and RUNPOD_POD_ID (auto-set by RunPod in most pod templates)
echo "Stopping pod..."
if command -v runpodctl &>/dev/null; then
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        runpodctl config --apiKey "$RUNPOD_API_KEY" 2>/dev/null
    fi
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "Stopping pod $RUNPOD_POD_ID via runpodctl..."
        runpodctl stop pod "$RUNPOD_POD_ID"
    else
        echo "RUNPOD_POD_ID not set. Attempting to find pod ID..."
        # Try to get pod ID from hostname or API
        POD_ID=$(runpodctl get pod 2>/dev/null | grep -o '[a-z0-9]\{24\}' | head -1 || true)
        if [ -n "$POD_ID" ]; then
            echo "Found pod $POD_ID, stopping..."
            runpodctl stop pod "$POD_ID"
        else
            echo "Could not determine pod ID. Stop the pod manually."
        fi
    fi
else
    echo "runpodctl not found. Stop the pod manually."
fi
