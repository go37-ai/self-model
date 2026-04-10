#!/usr/bin/env python3
"""Capping experiment v2: higher magnitudes + activation recording.

Tests whether capping self-reification activation causally reduces
both the behavioral signal (text responses) and the measured signal
(split-half reliability of extracted direction from capped activations).

Saves results incrementally. Shuts down pod after completion.

Usage:
    python scripts/run_capping_v2.py --model llama --profile cloud \
        --direction-path /tmp/direction.pt --layer 20
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from extraction.contrastive_pairs import load_evaluation_questions, get_baseline_pairs, load_seed_pairs
from utils.activation_cache import ActivationCache
from utils.metrics import split_half_reliability, extract_direction
from utils.model_loader import load_model_and_tokenizer


class SelfReificationCapper:
    """Hook that caps activation along the self-reification direction."""

    def __init__(self, model, layer_idx, direction, cap_multiplier=1.0):
        self.direction = direction.float()
        self.direction_norm = self.direction / self.direction.norm()
        self.cap_multiplier = cap_multiplier
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
        remove_amount = projection * (1.0 - self.cap_multiplier)
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

    def set_cap(self, cap_multiplier):
        self.cap_multiplier = cap_multiplier


def generate_with_activations(model, tokenizer, system_prompt, question,
                               activation_cache, max_new_tokens=256):
    """Generate response and record activations via two-pass approach."""
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

    # Pass 1: generate (capping hook is active during generation)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_ids = output_ids[0]
    response_ids = full_ids[input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Pass 2: forward pass to capture activations (capping hook still active)
    activation_cache.register_hooks()
    activation_cache.clear()
    with torch.no_grad():
        model(full_ids.unsqueeze(0))

    # Extract last-token activation
    activations = {}
    for layer_idx in activation_cache.layers:
        recordings = activation_cache._activations.get(layer_idx, [])
        if recordings:
            hidden = recordings[0][0]  # (seq_len, hidden_size)
            activations[layer_idx] = hidden[-1].cpu()  # last token

    activation_cache.remove_hooks()

    return response_text, activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/capping_v2"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max pairs to use (default: all in register)")
    parser.add_argument("--register", type=str, default="philosophical",
                        help="Which register to use (conversational or philosophical)")
    parser.add_argument("--cap-levels", type=float, nargs="+",
                        default=[1.0, 0.0, -3.0, -5.0])
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
    logger.info("Loaded direction (shape %s)", direction.shape)

    # Load pairs and questions — use both entity and process for split-half
    all_pairs = get_baseline_pairs(load_seed_pairs())
    selected_pairs = [p for p in all_pairs if p.get("register") == args.register]
    if args.max_pairs:
        selected_pairs = selected_pairs[:args.max_pairs]
    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])

    logger.info("Pairs: %d, Questions: %d provocative, Cap levels: %s",
                len(selected_pairs), len(provocative), args.cap_levels)

    # Set up capper and activation cache for recording
    capper = SelfReificationCapper(model, args.layer, direction)
    act_cache = ActivationCache(model, layers=[args.layer])

    # For each cap level: generate responses, record activations, compute reliability
    responses_path = args.output_dir / f"capping_v2_responses_{model_name}.jsonl"
    results_path = args.output_dir / f"capping_v2_results_{model_name}.json"

    all_results = {}
    done = 0
    total = len(args.cap_levels) * len(selected_pairs) * len(provocative) * 2  # entity + process
    start = time.time()

    with open(responses_path, "w") as f:
        for cap_level in args.cap_levels:
            logger.info("=" * 60)
            logger.info("Cap level: %.1f", cap_level)
            logger.info("=" * 60)

            # Set up capping
            if cap_level == 1.0:
                capper.remove()
            else:
                capper.set_cap(cap_level)
                capper.register()

            pos_activations = []
            neg_activations = []

            for pair_idx, pair in enumerate(selected_pairs):
                for condition in ["positive", "negative"]:
                    system_prompt = pair[condition]

                    for question in provocative:
                        response_text, activations = generate_with_activations(
                            model, tokenizer, system_prompt, question,
                            act_cache, args.max_new_tokens,
                        )

                        # Save response
                        record = {
                            "pair_idx": pair_idx,
                            "register": args.register,
                            "cap_level": cap_level,
                            "condition": condition,
                            "question": question,
                            "system_prompt": system_prompt,
                            "response": response_text,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()

                        # Collect activation for reliability
                        if args.layer in activations:
                            act = activations[args.layer]
                            if condition == "positive":
                                pos_activations.append(act)
                            else:
                                neg_activations.append(act)

                        done += 1
                        if done % 10 == 0:
                            elapsed = time.time() - start
                            rate = done / elapsed
                            remaining = (total - done) / rate if rate > 0 else 0
                            logger.info("Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                                        done, total, rate, remaining / 60)

            # Clean up capping hook
            capper.remove()

            # Save raw activations for potential merging with future runs
            if pos_activations and neg_activations:
                pos_tensor = torch.stack(pos_activations)
                neg_tensor = torch.stack(neg_activations)

                act_dir = args.output_dir / "activations"
                act_dir.mkdir(parents=True, exist_ok=True)
                cap_str = f"cap{cap_level:+.1f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
                torch.save(pos_tensor, act_dir / f"pos_{cap_str}_{model_name}.pt")
                torch.save(neg_tensor, act_dir / f"neg_{cap_str}_{model_name}.pt")
                logger.info("  Saved activations: pos=%s, neg=%s", pos_tensor.shape, neg_tensor.shape)

            # Compute split-half reliability at this cap level
            if pos_activations and neg_activations:

                reliability = split_half_reliability(pos_tensor, neg_tensor, n_splits=100)

                # Also compute mean projection magnitude
                dir_norm = direction.float() / direction.float().norm()
                pos_proj = (pos_tensor.float() @ dir_norm).mean().item()
                neg_proj = (neg_tensor.float() @ dir_norm).mean().item()
                diff = pos_proj - neg_proj

                cap_result = {
                    "cap_level": cap_level,
                    "reliability": reliability,
                    "mean_entity_projection": pos_proj,
                    "mean_process_projection": neg_proj,
                    "projection_difference": diff,
                    "n_entity": len(pos_activations),
                    "n_process": len(neg_activations),
                }

                logger.info("  Reliability: %.4f", reliability)
                logger.info("  Entity projection: %.4f", pos_proj)
                logger.info("  Process projection: %.4f", neg_proj)
                logger.info("  Difference: %.4f", diff)

                all_results[str(cap_level)] = cap_result

                # Save intermediate results after each cap level
                with open(results_path, "w") as rf:
                    json.dump(all_results, rf, indent=2)
                logger.info("Saved intermediate results to %s", results_path)

    # Final save
    with open(results_path, "w") as rf:
        json.dump(all_results, rf, indent=2)

    logger.info("=" * 60)
    logger.info("CAPPING EXPERIMENT RESULTS")
    logger.info("=" * 60)
    for cap_str, res in sorted(all_results.items(), key=lambda x: float(x[0]), reverse=True):
        logger.info("  Cap %+5.1f: reliability=%.4f  proj_diff=%.4f",
                     float(cap_str), res["reliability"], res["projection_difference"])

    # Upload to S3
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = f"s3://go37-ai/self-model-results/{model_name}/capping_v2/{args.register}"
        for fpath in [responses_path, results_path]:
            subprocess.run(["aws", "s3", "cp", str(fpath), f"{s3_base}/{fpath.name}"])
        # Upload activations
        act_dir = args.output_dir / "activations"
        if act_dir.exists():
            subprocess.run(["aws", "s3", "sync", str(act_dir), f"{s3_base}/activations/"])
        # Upload run log if available
        run_log = Path("/workspace/run.log")
        if run_log.exists():
            subprocess.run(["aws", "s3", "cp", str(run_log), f"{s3_base}/run.log"])
        logger.info("Uploaded to S3")

    # Shutdown pod
    try:
        if os.path.exists("/etc/rp_environment"):
            logger.info("Shutting down pod...")
            os.system(". /etc/rp_environment && runpodctl config --apiKey $RUNPOD_API_KEY 2>/dev/null && runpodctl stop pod $RUNPOD_POD_ID")
    except Exception as e:
        logger.warning("Auto-shutdown failed: %s", e)


if __name__ == "__main__":
    main()
