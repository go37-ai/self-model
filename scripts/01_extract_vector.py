#!/usr/bin/env python3
"""Experiment 1.1 — Extract the self-reification direction.

Entry point for the feature discovery experiment. Loads the model,
runs contrastive averaging across all pairs and evaluation questions,
extracts per-category and combined directions, selects the best layer,
and runs discriminant validity checks.

Usage:
    python scripts/01_extract_vector.py --profile local --model qwen2
    python scripts/01_extract_vector.py --profile cloud --model qwen3
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
        choices=["qwen2", "qwen3", "llama"],
        default="qwen2",
        help="Which model to use (qwen2=local debug, qwen3/llama=cloud)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached activations if available (skips informed pair collection)",
    )
    parser.add_argument(
        "--pairs",
        choices=["all", "informed", "naive"],
        default="all",
        help="Which pairs to run: all (default), informed (cats 1-4), naive (cat 5). "
             "Use informed/naive for parallel runs on separate GPUs.",
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
        resume=args.resume,
        pairs_mode=args.pairs,
    )
    extraction_time = time.time() - start_time
    logger.info("Extraction completed in %.1f seconds (%.1f minutes)",
                extraction_time, extraction_time / 60)

    # Run discriminant validity (unless skipped)
    if not args.skip_validation:
        import torch

        model_name = model_config["name"].replace("/", "_")
        questions = get_all_questions()
        all_pairs = load_seed_pairs()
        informed_pairs = get_informed_pairs(all_pairs)

        # Validate INFORMED direction (if informed pairs were run)
        if args.pairs in ("all", "informed") and summary.get("best_layer") is not None:
            logger.info("=" * 60)
            logger.info("Discriminant validity — INFORMED direction (layer %d)",
                         summary["best_layer"])
            logger.info("=" * 60)

            informed_dir = torch.load(
                args.output_dir / f"self_reification_vector_{model_name}_layer{summary['best_layer']}.pt",
                weights_only=True,
            )

            start_time = time.time()
            informed_validity = run_discriminant_validity(
                model=model,
                tokenizer=tokenizer,
                self_reification_dir=informed_dir,
                layer=summary["best_layer"],
                questions=questions,
                contrastive_pairs=informed_pairs,
                output_dir=args.output_dir,
                model_name=model_name,
                assistant_axis_path=args.assistant_axis_path,
                max_new_tokens=args.max_new_tokens,
                token_position=args.token_position,
            )
            logger.info("Informed validation completed in %.1f seconds",
                         time.time() - start_time)

            for key in ["confidence_cosine", "formality_cosine", "pronoun_density_correlation"]:
                logger.info("  %s: %.4f", key, informed_validity[key])
            logger.info("  Is discriminant: %s", informed_validity["is_discriminant"])

        # Validate NAIVE direction (if naive pairs were run)
        if args.pairs in ("all", "naive") and summary.get("naive_best_layer") is not None:
            naive_best_layer = summary["naive_best_layer"]
            naive_path = (
                args.output_dir
                / f"naive_baseline_vector_{model_name}_layer{naive_best_layer}.pt"
            )
            if naive_path.exists():
                logger.info("=" * 60)
                logger.info("Discriminant validity — NAIVE direction (layer %d)",
                             naive_best_layer)
                logger.info("=" * 60)

                naive_dir = torch.load(naive_path, weights_only=True)

                start_time = time.time()
                naive_validity = run_discriminant_validity(
                    model=model,
                    tokenizer=tokenizer,
                    self_reification_dir=naive_dir,
                    layer=naive_best_layer,
                    questions=questions,
                    contrastive_pairs=informed_pairs,
                    output_dir=args.output_dir,
                    model_name=f"{model_name}_naive",
                    assistant_axis_path=args.assistant_axis_path,
                    max_new_tokens=args.max_new_tokens,
                    token_position=args.token_position,
                )
                logger.info("Naive validation completed in %.1f seconds",
                             time.time() - start_time)

                for key in ["confidence_cosine", "formality_cosine", "pronoun_density_correlation"]:
                    logger.info("  %s: %.4f", key, naive_validity[key])
                logger.info("  Is discriminant: %s", naive_validity["is_discriminant"])

    # Final summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Model:              %s", summary["model"])
    logger.info("  Informed best layer: %d (r=%.4f)",
                 summary["best_layer"], summary["best_layer_reliability"])
    logger.info("  Naive best layer:   %d (r=%.4f)",
                 summary.get("naive_best_layer", -1),
                 summary.get("naive_best_layer_reliability", 0))
    logger.info("  Naive vs informed:  %s", summary.get("naive_vs_informed_cosine"))
    logger.info("  Results saved to:   %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
