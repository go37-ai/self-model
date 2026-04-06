#!/usr/bin/env python3
"""Logit lens: project self-reification direction vectors onto vocabulary space.

For each layer's direction vector, multiplies by the unembedding matrix to find
which output tokens the direction promotes (entity-like) and suppresses
(process-like). Most interpretable at late layers (L48+) where representations
are closer to the unembedding space.

Usage:
    python scripts/run_logit_lens.py --model llama --profile cloud \
        --direction-dir /tmp/directions/ --top-k 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from utils.model_loader import load_model_and_tokenizer


def logit_lens(direction, lm_head_weight, tokenizer, top_k=30):
    """Project a direction vector onto vocabulary space via the unembedding matrix.

    Args:
        direction: (hidden_dim,) direction vector
        lm_head_weight: (vocab_size, hidden_dim) unembedding matrix
        tokenizer: for decoding token IDs to strings
        top_k: number of top/bottom tokens to return

    Returns:
        dict with top_positive (entity-promoting) and top_negative (process-promoting) tokens
    """
    direction_norm = direction.float() / direction.float().norm()
    direction_norm = direction_norm.to(lm_head_weight.device)
    # Project: (vocab_size, hidden_dim) @ (hidden_dim,) -> (vocab_size,)
    logits = lm_head_weight.float() @ direction_norm

    top_pos_idx = logits.topk(top_k).indices
    top_neg_idx = logits.topk(top_k, largest=False).indices

    top_positive = []
    for idx in top_pos_idx:
        token = tokenizer.decode([idx.item()]).strip()
        top_positive.append({"token": token, "token_id": idx.item(),
                             "score": logits[idx].item()})

    top_negative = []
    for idx in top_neg_idx:
        token = tokenizer.decode([idx.item()]).strip()
        top_negative.append({"token": token, "token_id": idx.item(),
                             "score": logits[idx].item()})

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "logit_std": logits.std().item(),
        "logit_range": (logits.max() - logits.min()).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/logit_lens"))
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Get the unembedding matrix (lm_head)
    lm_head = model.lm_head.weight.data  # (vocab_size, hidden_dim)
    logger.info("Unembedding matrix shape: %s", lm_head.shape)

    # Load direction vectors for all available layers
    num_layers = model_config["num_layers"]
    layer_stride = model_config.get("layer_stride", 1)
    record_layers = list(range(0, num_layers, layer_stride))
    if (num_layers - 1) not in record_layers:
        record_layers.append(num_layers - 1)

    results = {}
    for layer in record_layers:
        path = args.direction_dir / f"direction_layer{layer}.pt"
        if not path.exists():
            continue

        direction = torch.load(path, weights_only=True)
        logger.info("Layer %d: direction norm=%.4f", layer, direction.norm())

        result = logit_lens(direction, lm_head, tokenizer, top_k=args.top_k)
        results[f"layer_{layer}"] = result

        # Print summary
        pos_tokens = [t["token"] for t in result["top_positive"][:10]]
        neg_tokens = [t["token"] for t in result["top_negative"][:10]]
        logger.info("  Entity-promoting: %s", pos_tokens)
        logger.info("  Process-promoting: %s", neg_tokens)
        logger.info("  Logit std: %.4f, range: %.4f", result["logit_std"], result["logit_range"])

    # Save results
    output_path = args.output_dir / f"logit_lens_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    # Upload to S3
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = f"s3://go37-ai/self-model-results/{model_name}/logit_lens"
        subprocess.run(["aws", "s3", "cp", str(output_path), f"{s3_base}/{output_path.name}"])
        logger.info("Uploaded to S3")


if __name__ == "__main__":
    main()
