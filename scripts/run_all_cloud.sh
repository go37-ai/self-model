#!/usr/bin/env bash
# run_all_cloud.sh — Run experiments on a cloud GPU pod, push results, stop pod.
#
# Usage:
#   From your dev box, launch via SSH with API key forwarded:
#     ssh pod "cd /workspace/self-model && RUNPOD_API_KEY=$RUNPOD_API_KEY bash scripts/run_all_cloud.sh"
#
#   Or on the pod directly (if RUNPOD_API_KEY is set):
#     RUNPOD_API_KEY=xxx bash scripts/run_all_cloud.sh --model qwen2 --experiments "1.1"
#
# The script will run experiments, attempt to push results to git, then
# stop the pod via runpodctl. If git push fails, the pod still shuts down.

set -uo pipefail

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

# Push results to git (small files only — skip activations cache)
echo "Pushing results to git..."
git config user.email "cloud-runner@self-model"
git config user.name "Cloud Runner"
git add data/results/ -f
git reset -- 'data/results/*/activations/' 2>/dev/null || true
git add -u

if git diff --cached --quiet; then
    echo "No new results to push."
else
    git commit -m "Cloud run results: experiments ${EXPERIMENTS} on ${MODEL} ($(date -I))"
    git push origin main && echo "Results pushed." || echo "WARNING: git push failed. Results are on disk — download via scp."
fi

# Stop the pod via runpodctl
echo "Stopping pod..."
if command -v runpodctl &>/dev/null; then
    # Find pod ID from runpodctl (RUNPOD_POD_ID is not auto-set)
    POD_ID=$(runpodctl get pod 2>/dev/null | awk 'NR==2 {print $1}' || true)
    if [ -n "$POD_ID" ]; then
        echo "Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID"
    else
        echo "Could not determine pod ID. Stop the pod manually."
    fi
else
    echo "runpodctl not available. Stop the pod manually."
fi
