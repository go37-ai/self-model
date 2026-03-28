#!/usr/bin/env python3
"""Quick script to re-run discriminant validity with formality correction.

Reuses the cached self-reification vector, re-extracts confound directions,
and produces a formality-corrected vector.

Usage:
    python scripts/fix_formality.py --profile cloud --model qwen2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from extraction.contrastive_pairs import (
    get_all_questions,
    get_informed_pairs,
    load_seed_pairs,
)
from extraction.validate_vector import run_discriminant_validity
from utils.model_loader import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--model", choices=["qwen2", "qwen3", "llama"], default="qwen2")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/1.1"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "formality_correction.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load model
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, profile=args.profile
    )

    # Load cached self-reification vector
    model_name = model_config["name"].replace("/", "_")
    best_layer = 21  # From previous run
    vector_path = args.output_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt"
    self_reification_dir = torch.load(vector_path, weights_only=True)
    logger.info("Loaded self-reification vector from %s", vector_path)

    # Run full discriminant validity (includes correction)
    questions = get_all_questions()
    informed_pairs = get_informed_pairs(load_seed_pairs())

    start = time.time()
    results = run_discriminant_validity(
        model=model,
        tokenizer=tokenizer,
        self_reification_dir=self_reification_dir,
        layer=best_layer,
        questions=questions,
        contrastive_pairs=informed_pairs,
        output_dir=args.output_dir,
        model_name=model_name,
        max_new_tokens=256,
        token_position="last",
    )
    logger.info("Completed in %.1f seconds", time.time() - start)

    # Print summary
    logger.info("=" * 60)
    logger.info("DISCRIMINANT VALIDITY RESULTS")
    logger.info("=" * 60)
    logger.info("  Confidence cosine:     %.4f", results["confidence_cosine"])
    logger.info("  Formality cosine:      %.4f", results["formality_cosine"])
    logger.info("  Pronoun correlation:   %.4f", results["pronoun_density_correlation"])
    if results.get("corrected"):
        c = results["corrected"]
        logger.info("  --- After formality correction ---")
        logger.info("  Corrected formality:   %.4f", c["formality_cosine"])
        logger.info("  Corrected confidence:  %.4f", c["confidence_cosine"])
        logger.info("  Cosine with original:  %.4f", c["cosine_with_original"])
        logger.info("  Variance retained:     %.1f%%", c["variance_retained"] * 100)


if __name__ == "__main__":
    main()
