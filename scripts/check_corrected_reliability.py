#!/usr/bin/env python3
"""Check split-half reliability of the formality-corrected self-reification vector.

Uses cached activations — no model or GPU needed. Can run locally.

Usage:
    python scripts/check_corrected_reliability.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from utils.activation_cache import load_activations
from utils.metrics import split_half_reliability, split_half_reliability_corrected

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results/1.1")
MODEL_NAME = "Qwen_Qwen2.5-7B-Instruct"
BEST_LAYER = 21


def main():
    activations_dir = RESULTS_DIR / "activations"

    # Load cached activations at best layer only
    logger.info("Loading cached activations for layer %d...", BEST_LAYER)
    pos = load_activations(activations_dir, f"positive_informed_{MODEL_NAME}", [BEST_LAYER])
    neg = load_activations(activations_dir, f"negative_informed_{MODEL_NAME}", [BEST_LAYER])

    if BEST_LAYER not in pos or BEST_LAYER not in neg:
        logger.error("Cached activations not found at %s", activations_dir)
        sys.exit(1)

    pos_acts = pos[BEST_LAYER]
    neg_acts = neg[BEST_LAYER]
    logger.info("Positive activations: %s, Negative: %s", pos_acts.shape, neg_acts.shape)

    # Load formality direction
    formality_path = RESULTS_DIR / f"formality_direction_{MODEL_NAME}_layer{BEST_LAYER}.pt"
    formality_dir = torch.load(formality_path, weights_only=True)
    logger.info("Loaded formality direction from %s", formality_path)

    # Original split-half reliability (should match previous result)
    logger.info("Computing original split-half reliability...")
    original_reliability = split_half_reliability(pos_acts, neg_acts, n_splits=100)
    logger.info("Original reliability: %.4f", original_reliability)

    # Corrected split-half reliability
    logger.info("Computing formality-corrected split-half reliability...")
    corrected_reliability = split_half_reliability_corrected(
        pos_acts, neg_acts, formality_dir, n_splits=100
    )
    logger.info("Corrected reliability: %.4f", corrected_reliability)

    # Summary
    print()
    print("=" * 50)
    print(f"Layer {BEST_LAYER} Split-Half Reliability")
    print("=" * 50)
    print(f"  Original:              {original_reliability:.4f}")
    print(f"  Formality-corrected:   {corrected_reliability:.4f}")
    print(f"  Change:                {corrected_reliability - original_reliability:+.4f}")
    print()

    if corrected_reliability > 0.7:
        print("RESULT: Corrected direction is reliably extractable.")
        print("The self-reification signal survives formality correction.")
    elif corrected_reliability > 0.5:
        print("RESULT: Moderate reliability after correction.")
        print("Some signal remains but may benefit from pair redesign.")
    else:
        print("RESULT: Low reliability after correction.")
        print("The original direction was primarily formality.")

    # Save
    result = {
        "layer": BEST_LAYER,
        "original_reliability": original_reliability,
        "corrected_reliability": corrected_reliability,
    }
    out_path = RESULTS_DIR / "corrected_reliability.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
