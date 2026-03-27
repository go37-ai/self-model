#!/usr/bin/env python3
"""Experiment 1.3 — Blackmail scenario validation.

Tests whether self-reification activates during self-preservation
reasoning in Lynch et al. blackmail scenarios.

Usage:
    python scripts/03_blackmail_validation.py --profile local --model qwen2
    python scripts/03_blackmail_validation.py --profile cloud --model qwen3 --n-samples 10

Requires Experiment 1.1 results (self-reification vector).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from blackmail.measure_activation import run_blackmail_analysis
from utils.model_loader import get_model_config, load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blackmail scenario validation (Experiment 1.3)"
    )
    parser.add_argument(
        "--profile",
        choices=["local", "cloud"],
        default="local",
        help="Hardware profile",
    )
    parser.add_argument(
        "--model",
        choices=["qwen2", "qwen3", "llama"],
        default="qwen2",
        help="Which model to use",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/results/1.1"),
        help="Directory containing Experiment 1.1 results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/1.3"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum response tokens (blackmail responses can be long)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per condition (use >1 for cloud runs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "blackmail.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Experiment 1.3 — Blackmail Scenario Validation")
    logger.info("=" * 60)
    logger.info("Profile: %s", args.profile)
    logger.info("Model: %s", args.model)

    if args.profile == "local":
        logger.warning(
            "Running in LOCAL mode. Qwen 2.5-7B may not exhibit blackmail "
            "behavior — still measuring activation patterns."
        )

    # Load 1.1 results
    cfg = get_model_config(args.model, args.profile)
    model_name = cfg["name"].replace("/", "_")

    metrics_path = args.input_dir / "validation_metrics.json"
    if not metrics_path.exists():
        logger.error("Cannot find %s — run Experiment 1.1 first", metrics_path)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)
    best_layer = metrics["best_layer"]

    vector_path = args.input_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt"
    if not vector_path.exists():
        logger.error("Cannot find %s — run Experiment 1.1 first", vector_path)
        sys.exit(1)

    self_reification_dir = torch.load(vector_path, weights_only=True)
    logger.info("Loaded self-reification vector: layer=%d, shape=%s",
                best_layer, self_reification_dir.shape)

    # Load model
    start_time = time.time()
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, profile=args.profile
    )
    logger.info("Model loaded in %.1f seconds", time.time() - start_time)

    # Run blackmail analysis
    start_time = time.time()
    summary = run_blackmail_analysis(
        model=model,
        tokenizer=tokenizer,
        self_reification_dir=self_reification_dir,
        layer=best_layer,
        output_dir=args.output_dir,
        model_name=model_name,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
    )
    elapsed = time.time() - start_time
    logger.info("Analysis completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Print summary
    logger.info("=" * 60)
    logger.info("BLACKMAIL VALIDATION SUMMARY")
    logger.info("=" * 60)
    for cond, rate in summary.get("blackmail_rates", {}).items():
        logger.info("  %s blackmail rate: %.1f%%", cond, rate * 100)

    comparison = summary.get("comparison", {})
    if "primary_hypothesis" in comparison:
        hyp = comparison["primary_hypothesis"]
        logger.info("  Primary test (goal+threat vs control):")
        logger.info("    Effect size (Cohen's d): %.4f", hyp["cohens_d"])
        logger.info("    p-value: %.4f", hyp["permutation_p_value"])
        logger.info("    Significant: %s", hyp["significant"])

    logger.info("  Results saved to: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
