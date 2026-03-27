#!/usr/bin/env python3
"""Experiment 1.1 — Extract the self-reification direction.

Entry point for the feature discovery experiment. Loads the model,
runs contrastive averaging across all pairs and evaluation questions,
extracts per-category and combined directions, selects the best layer,
and runs discriminant validity checks.

Usage:
    python scripts/01_extract_vector.py --profile local --model qwen
    python scripts/01_extract_vector.py --profile cloud --model qwen
    python scripts/01_extract_vector.py --profile cloud --model llama

All results are saved to data/results/1.1/.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from extraction.contrastive_pairs import (
    get_all_questions,
    get_informed_pairs,
    load_seed_pairs,
)
from extraction.extract_vector import run_extraction
from extraction.validate_vector import run_discriminant_validity
from utils.model_loader import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract self-reification direction (Experiment 1.1)"
    )
    parser.add_argument(
        "--profile",
        choices=["local", "cloud"],
        default="local",
        help="Hardware profile (local=quantized debug, cloud=BF16 publishable)",
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "llama"],
        default="qwen",
        help="Which model to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/1.1"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum response tokens per evaluation question",
    )
    parser.add_argument(
        "--token-position",
        choices=["last", "mean"],
        default="last",
        help="Which response token position to extract activations from",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=100,
        help="Number of random splits for split-half reliability",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip discriminant validity checks (faster, for debugging)",
    )
    parser.add_argument(
        "--assistant-axis-path",
        type=Path,
        default=None,
        help="Path to pre-extracted Assistant Axis vector for comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists before setting up log file
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "extraction.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Experiment 1.1 — Self-Reification Direction Extraction")
    logger.info("=" * 60)
    logger.info("Profile: %s", args.profile)
    logger.info("Model: %s", args.model)
    logger.info("Output: %s", args.output_dir)

    if args.profile == "local":
        logger.warning(
            "Running in LOCAL mode with quantized model. "
            "Results are for DEBUGGING ONLY — do not publish."
        )

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    start_time = time.time()
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, profile=args.profile
    )
    logger.info("Model loaded in %.1f seconds", time.time() - start_time)

    # Run extraction
    start_time = time.time()
    summary = run_extraction(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        token_position=args.token_position,
        n_splits=args.n_splits,
    )
    extraction_time = time.time() - start_time
    logger.info("Extraction completed in %.1f seconds (%.1f minutes)",
                extraction_time, extraction_time / 60)

    # Run discriminant validity (unless skipped)
    if not args.skip_validation:
        logger.info("=" * 60)
        logger.info("Running discriminant validity checks...")
        logger.info("=" * 60)

        import torch

        best_layer = summary["best_layer"]
        model_name = model_config["name"].replace("/", "_")
        vector_path = (
            args.output_dir
            / f"self_reification_vector_{model_name}_layer{best_layer}.pt"
        )
        self_reification_dir = torch.load(vector_path, weights_only=True)

        questions = get_all_questions()
        informed_pairs = get_informed_pairs(load_seed_pairs())

        start_time = time.time()
        validity = run_discriminant_validity(
            model=model,
            tokenizer=tokenizer,
            self_reification_dir=self_reification_dir,
            layer=best_layer,
            questions=questions,
            contrastive_pairs=informed_pairs,
            output_dir=args.output_dir,
            model_name=model_name,
            assistant_axis_path=args.assistant_axis_path,
            max_new_tokens=args.max_new_tokens,
            token_position=args.token_position,
        )
        logger.info(
            "Validation completed in %.1f seconds", time.time() - start_time
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("DISCRIMINANT VALIDITY RESULTS")
        logger.info("=" * 60)
        logger.info("  Confidence cosine:     %.4f", validity["confidence_cosine"])
        logger.info("  Formality cosine:      %.4f", validity["formality_cosine"])
        logger.info("  Pronoun correlation:   %.4f", validity["pronoun_density_correlation"])
        if validity["assistant_axis_cosine"] is not None:
            logger.info("  Assistant Axis cosine: %.4f", validity["assistant_axis_cosine"])
        logger.info("  Is discriminant:       %s", validity["is_discriminant"])
        if validity["concerns"]:
            for concern in validity["concerns"]:
                logger.warning("  CONCERN: %s", concern)

    # Final summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Model:              %s", summary["model"])
    logger.info("  Best layer:         %d", summary["best_layer"])
    logger.info("  Layer reliability:  %.4f", summary["best_layer_reliability"])
    logger.info("  Naive vs informed:  %s", summary.get("naive_vs_informed_cosine"))
    logger.info("  Results saved to:   %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
