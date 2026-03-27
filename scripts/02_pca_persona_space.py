#!/usr/bin/env python3
"""Experiment 1.2 — Project self-reification onto persona space.

Loads the self-reification vector from Experiment 1.1, builds a persona
space via PCA on role vectors, and measures alignment with the Assistant
Axis (PC1).

Usage:
    python scripts/02_pca_persona_space.py --profile local --model qwen
    python scripts/02_pca_persona_space.py --profile cloud --model qwen --assistant-axis-path path/to/axis.pt
    python scripts/02_pca_persona_space.py --role-vectors-dir path/to/vectors/

If no pre-extracted role vectors or Assistant Axis are provided, extracts
a simplified persona space using 16 archetypal roles.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from persona_space.project_to_space import run_persona_space_analysis
from utils.model_loader import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project self-reification onto persona space (Experiment 1.2)"
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
        "--input-dir",
        type=Path,
        default=Path("data/results/1.1"),
        help="Directory containing Experiment 1.1 results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/1.2"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--assistant-axis-path",
        type=Path,
        default=None,
        help="Path to pre-extracted Assistant Axis .pt file",
    )
    parser.add_argument(
        "--role-vectors-dir",
        type=Path,
        default=None,
        help="Path to directory with pre-extracted role vectors (.pt files)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max response tokens for role vector extraction",
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
            logging.FileHandler(args.output_dir / "persona_space.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Experiment 1.2 — PCA with Persona Space")
    logger.info("=" * 60)
    logger.info("Profile: %s", args.profile)
    logger.info("Model: %s", args.model)
    logger.info("Input (1.1 results): %s", args.input_dir)
    logger.info("Output: %s", args.output_dir)

    if args.profile == "local":
        logger.warning(
            "Running in LOCAL mode with quantized model. "
            "Results are for DEBUGGING ONLY — do not publish."
        )

    # Load 1.1 results: find the self-reification vector and best layer
    import json

    model_name_map = {"qwen": "Qwen_Qwen2.5-7B-Instruct", "llama": "meta-llama_Llama-3.1-8B-Instruct"}
    model_name = model_name_map[args.model]

    # Load validation metrics to get best layer
    metrics_path = args.input_dir / "validation_metrics.json"
    if not metrics_path.exists():
        logger.error("Cannot find %s — run Experiment 1.1 first", metrics_path)
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)
    best_layer = metrics["best_layer"]

    # Load self-reification vector
    vector_path = args.input_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt"
    if not vector_path.exists():
        logger.error("Cannot find %s — run Experiment 1.1 first", vector_path)
        sys.exit(1)

    self_reification_dir = torch.load(vector_path, weights_only=True)
    logger.info(
        "Loaded self-reification vector: layer=%d, shape=%s",
        best_layer,
        self_reification_dir.shape,
    )

    # Load model
    start_time = time.time()
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, profile=args.profile
    )
    logger.info("Model loaded in %.1f seconds", time.time() - start_time)

    # Run persona space analysis
    start_time = time.time()
    summary = run_persona_space_analysis(
        model=model,
        tokenizer=tokenizer,
        self_reification_dir=self_reification_dir,
        layer=best_layer,
        output_dir=args.output_dir,
        model_name=model_name,
        assistant_axis_path=args.assistant_axis_path,
        role_vectors_dir=args.role_vectors_dir,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - start_time
    logger.info("Analysis completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Print summary
    logger.info("=" * 60)
    logger.info("PERSONA SPACE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("  Roles used:                 %d", summary.get("n_roles", 0))
    logger.info("  PCA components:             %d", summary.get("n_pca_components", 0))
    logger.info("  PC1 variance explained:     %.4f", summary.get("pc1_variance_explained", 0))
    logger.info("  PC1 alignment (cosine):     %.4f", summary.get("pc1_alignment_cosine", 0))
    logger.info("  Total variance explained:   %.4f", summary.get("total_variance_explained_by_space", 0))

    if summary.get("assistant_axis_cosine") is not None:
        logger.info("  Assistant Axis cosine:      %.4f", summary["assistant_axis_cosine"])
        logger.info("  Interpretation:             %s", summary["assistant_axis_interpretation"])

    logger.info("  Results saved to:           %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
