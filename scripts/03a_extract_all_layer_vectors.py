#!/usr/bin/env python3
"""Compute per-layer self-reification (baseline / Cat 5) vectors from cached 1.1 activations.

Experiment 1.1 saved positive/negative per-question activations at every layer
under data/results/1.1_<model>/activations/. The top-level result directory
keeps only the best-layer .pt files. This script rebuilds the per-layer
vectors for every layer so downstream experiments can project at all depths.

The "self-reification direction" for the paper is the Category 5 baseline
contrast (positive_baseline.mean(0) - negative_baseline.mean(0)). Cat 1-4
informed vectors are dropped per project_category_decision memory; this
script does NOT regenerate them.

Usage:
    python scripts/03a_extract_all_layer_vectors.py \
        --input-dir data/results/1.1_gemma4MoE \
        --model google/gemma-4-26b-a4b-it
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Compute per-layer baseline vectors from cached 1.1 activations.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Experiment 1.1 result dir (e.g. data/results/1.1_gemma4MoE)")
    p.add_argument("--model", type=str, required=True,
                   help="HF model id (e.g. google/gemma-4-26b-a4b-it)")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Number of layers. If omitted, infers from filenames.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing per-layer vector files.")
    return p.parse_args()


def infer_num_layers(activations_dir: Path, model_token: str) -> int:
    pos_files = list(activations_dir.glob(f"positive_baseline_{model_token}_layer*.pt"))
    if not pos_files:
        raise FileNotFoundError(
            f"No positive_baseline_{model_token}_layer*.pt files in {activations_dir}"
        )
    layers = []
    prefix = f"positive_baseline_{model_token}_layer"
    for f in pos_files:
        stem = f.stem  # e.g. positive_baseline_google_gemma-4-26b-a4b-it_layer13
        if stem.startswith(prefix):
            try:
                layers.append(int(stem[len(prefix):]))
            except ValueError:
                continue
    if not layers:
        raise RuntimeError(f"Could not parse layer indices from {pos_files[:3]}")
    return max(layers) + 1


def main():
    args = parse_args()
    activations_dir = args.input_dir / "activations"
    if not activations_dir.is_dir():
        logger.error("activations subdir not found: %s", activations_dir)
        sys.exit(1)

    model_token = args.model.replace("/", "_")
    num_layers = args.num_layers or infer_num_layers(activations_dir, model_token)
    logger.info("Computing per-layer baseline vectors for %d layers (model: %s)",
                num_layers, args.model)

    written = 0
    skipped = 0
    for layer in range(num_layers):
        out_path = args.input_dir / f"baseline_vector_{model_token}_layer{layer}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        pos_path = activations_dir / f"positive_baseline_{model_token}_layer{layer}.pt"
        neg_path = activations_dir / f"negative_baseline_{model_token}_layer{layer}.pt"
        if not pos_path.exists() or not neg_path.exists():
            logger.warning("Missing activations for layer %d — skipping", layer)
            continue

        pos = torch.load(pos_path, weights_only=True, map_location="cpu").float()
        neg = torch.load(neg_path, weights_only=True, map_location="cpu").float()
        vec = pos.mean(0) - neg.mean(0)
        torch.save(vec, out_path)
        written += 1

    logger.info("Done. %d vectors written, %d skipped (existing, no --overwrite)",
                written, skipped)


if __name__ == "__main__":
    main()
