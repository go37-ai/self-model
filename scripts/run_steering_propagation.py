#!/usr/bin/env python3
"""Measure how steering at layer 20 propagates through the network.

Records activations at ALL layers (stride 4) during steering at layer 20
with coefficient 0.0. Compare against baseline activations from the main
extraction run to see whether the zeroed-out signal recovers at deeper layers.

Usage:
    python scripts/run_steering_propagation.py --model llama --profile cloud \
        --direction-path /tmp/direction.pt --steer-layer 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from extraction.contrastive_pairs import load_evaluation_questions, get_baseline_pairs, load_seed_pairs
from utils.activation_cache import ActivationCache
from utils.metrics import cosine_similarity as cos_sim
from utils.model_loader import load_model_and_tokenizer


class SelfReificationSteerer:
    """Hook that steers activation along the self-reification direction."""

    def __init__(self, model, layer_idx, direction, coefficient=0.0):
        self.direction = direction.float()
        self.direction_norm = self.direction / self.direction.norm()
        self.coefficient = coefficient
        self.hook = None

        cache = ActivationCache(model, layers=[layer_idx])
        self.layer_module = cache._layer_modules[layer_idx]

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            hidden = output.last_hidden_state

        direction_dev = self.direction_norm.to(hidden.device)
        projection = torch.einsum('...d,d->...', hidden.float(), direction_dev)
        remove_amount = projection * (1.0 - self.coefficient)
        hidden_modified = hidden.float() - remove_amount.unsqueeze(-1) * direction_dev
        hidden_modified = hidden_modified.to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return hidden_modified
        else:
            output.last_hidden_state = hidden_modified
            return output

    def register(self):
        self.remove()
        self.hook = self.layer_module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-path", type=Path, required=True)
    parser.add_argument("--steer-layer", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/steering_propagation"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Load direction
    direction = torch.load(args.direction_path, weights_only=True)
    direction_norm = direction.float() / direction.float().norm()
    logger.info("Loaded direction (shape %s), steering at layer %d", direction.shape, args.steer_layer)

    # Determine recording layers
    num_layers = model_config["num_layers"]
    layer_stride = model_config.get("layer_stride", 1)
    record_layers = list(range(0, num_layers, layer_stride))
    if (num_layers - 1) not in record_layers:
        record_layers.append(num_layers - 1)
    logger.info("Recording %d layers (stride %d)", len(record_layers), layer_stride)

    # Load pairs and questions
    all_pairs = get_baseline_pairs(load_seed_pairs())
    pairs = [p for p in all_pairs if p.get("register") == "conversational"][:args.max_pairs]
    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])

    logger.info("Pairs: %d, Questions: %d", len(pairs), len(provocative))
    total = len(pairs) * len(provocative) * 2
    logger.info("Total generations: %d", total)

    # Set up steerer and multi-layer activation cache
    steerer = SelfReificationSteerer(model, args.steer_layer, direction, coefficient=0.0)
    act_cache = ActivationCache(model, layers=record_layers)

    # Collect projections and raw activations at each layer for each condition
    layer_projections = {l: {"positive": [], "negative": []} for l in record_layers}
    layer_activations = {l: {"positive": [], "negative": []} for l in record_layers}

    steerer.register()
    done = 0
    start = time.time()

    for pair_idx, pair in enumerate(pairs):
        for condition in ["positive", "negative"]:
            system_prompt = pair[condition]

            for question in provocative:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                device = next(model.parameters()).device
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[1]

                # Pass 1: generate with steering active
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                full_ids = output_ids[0]

                # Pass 2: forward pass to capture activations at all layers
                # (steering hook is still active at steer_layer)
                act_cache.register_hooks()
                act_cache.clear()
                with torch.no_grad():
                    model(full_ids.unsqueeze(0))

                # Extract last-token projection at each layer
                for layer_idx in record_layers:
                    recordings = act_cache._activations.get(layer_idx, [])
                    if recordings:
                        hidden = recordings[0][0]  # (seq_len, hidden_size)
                        last_token = hidden[-1].float().cpu()
                        proj = torch.dot(last_token, direction_norm.cpu()).item()
                        layer_projections[layer_idx][condition].append(proj)
                        layer_activations[layer_idx][condition].append(last_token)

                act_cache.remove_hooks()

                done += 1
                if done % 10 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    remaining = (total - done) / rate if rate > 0 else 0
                    logger.info("Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                                done, total, rate, remaining / 60)

    steerer.remove()

    # Save raw activations per layer for merging across runs
    act_dir = args.output_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    for layer in record_layers:
        for condition in ["positive", "negative"]:
            acts = layer_activations[layer][condition]
            if acts:
                tensor = torch.stack(acts)
                cond_short = "pos" if condition == "positive" else "neg"
                torch.save(tensor, act_dir / f"{cond_short}_layer{layer}_{model_name}.pt")
    logger.info("Saved raw activations for %d layers to %s", len(record_layers), act_dir)

    # Compute mean projection and difference at each layer
    results = {}
    logger.info("\nLayer  Pos_proj  Neg_proj  Diff")
    logger.info("-" * 40)
    for layer in record_layers:
        pos = layer_projections[layer]["positive"]
        neg = layer_projections[layer]["negative"]
        if pos and neg:
            import numpy as np
            mean_pos = np.mean(pos)
            mean_neg = np.mean(neg)
            diff = mean_pos - mean_neg
            results[str(layer)] = {
                "mean_entity_projection": mean_pos,
                "mean_process_projection": mean_neg,
                "projection_difference": diff,
                "n_entity": len(pos),
                "n_process": len(neg),
            }
            logger.info(f"{layer:>5d}  {mean_pos:>8.4f}  {mean_neg:>8.4f}  {diff:>8.4f}")

    # Save results
    results_path = args.output_dir / f"steering_propagation_{model_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", results_path)

    # Upload to S3
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = f"s3://go37-ai/self-model-results/{model_name}/steering_propagation"
        subprocess.run(["aws", "s3", "cp", str(results_path), f"{s3_base}/{results_path.name}"])
        if act_dir.exists():
            subprocess.run(["aws", "s3", "sync", str(act_dir), f"{s3_base}/activations/"])
        logger.info("Uploaded to S3")

    # Shutdown pod
    try:
        if os.path.exists("/etc/rp_environment"):
            logger.info("Shutting down pod...")
            # Pre-configure runpodctl (may already be configured)
            os.system("mkdir -p /root/.runpod && touch /root/.runpod/config.toml")
            os.system(". /etc/rp_environment && runpodctl config --apiKey $RUNPOD_API_KEY 2>/dev/null && runpodctl stop pod $RUNPOD_POD_ID")
    except Exception as e:
        logger.warning("Auto-shutdown failed: %s", e)


if __name__ == "__main__":
    main()
