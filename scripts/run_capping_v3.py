#!/usr/bin/env python3
"""Capping experiment v3: one-sided capping at later layers.

Caps positive projections to zero (entity-side only), leaving negative
projections unchanged. Tests capping at L40, L72, and L40+L72.
Entity condition only. Records activations at all layers.
Saves response text. Logs cap firing stats.

Usage:
    python scripts/run_capping_v3.py --model llama --profile cloud \
        --direction-dir /tmp/directions/
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
from extraction.contrastive_pairs import load_evaluation_questions, get_naive_pairs, load_seed_pairs
from utils.activation_cache import ActivationCache
from utils.model_loader import load_model_and_tokenizer


class OneSidedCapper:
    """Hook that caps projections above a threshold at a target layer.

    Default threshold=0 caps positive projections to zero.
    Negative threshold (e.g. -2) pushes projections into process territory
    without full inversion.
    """

    def __init__(self, model, layer_idx, direction, threshold=0.0):
        self.direction = direction.float()
        self.direction_norm = self.direction / self.direction.norm()
        self.threshold = threshold
        self.hook = None
        self.cap_log = []
        self.layer_idx = layer_idx

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
        above_mask = projection > self.threshold

        # Log last-token projection before capping
        last_proj = projection[0, -1].item()

        if above_mask.any():
            capped = projection.clone()
            capped[above_mask] = self.threshold
            remove_amount = projection - capped
            hidden_modified = hidden.float() - remove_amount.unsqueeze(-1) * direction_dev
            hidden_modified = hidden_modified.to(hidden.dtype)

            fired = above_mask[0, -1].item()
            self.cap_log.append({"fired": fired, "projection": last_proj})

            if isinstance(output, tuple):
                return (hidden_modified,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return hidden_modified
            else:
                output.last_hidden_state = hidden_modified
                return output
        else:
            self.cap_log.append({"fired": False, "projection": last_proj})
            return output

    def register(self):
        self.remove()
        self.hook = self.layer_module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None

    def clear_log(self):
        self.cap_log = []

    def get_stats(self):
        if not self.cap_log:
            return {"fired": 0, "total": 0, "fire_rate": 0, "mean_projection": 0}
        fired = sum(1 for e in self.cap_log if e["fired"])
        projs = [e["projection"] for e in self.cap_log if e["fired"]]
        return {
            "fired": fired,
            "total": len(self.cap_log),
            "fire_rate": fired / len(self.cap_log),
            "mean_projection_when_fired": np.mean(projs) if projs else 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/capping_v3"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--cap-threshold", type=float, default=0.0,
                        help="Cap projection threshold. 0=zero positive projections, -2=push to -2.")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Which cap conditions to run (e.g., cap_L40 cap_all_from_L4). Default: all.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")

    # Recording layers
    num_layers = model_config["num_layers"]
    layer_stride = model_config.get("layer_stride", 1)
    record_layers = list(range(0, num_layers, layer_stride))
    if (num_layers - 1) not in record_layers:
        record_layers.append(num_layers - 1)
    logger.info("Recording %d layers", len(record_layers))

    # Load all available direction vectors
    directions = {}
    for layer in record_layers:
        path = args.direction_dir / f"direction_layer{layer}.pt"
        if path.exists():
            directions[layer] = torch.load(path, weights_only=True)
            logger.info("Loaded direction for layer %d (norm=%.4f)", layer, directions[layer].norm())
    logger.info("Loaded directions for %d layers", len(directions))

    # Load pairs and questions
    all_pairs = get_naive_pairs(load_seed_pairs())
    conv_pairs = [p for p in all_pairs if p.get("register") == "conversational"]
    if args.max_pairs:
        conv_pairs = conv_pairs[:args.max_pairs]
    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])

    # Capping conditions
    # Build cap conditions — all available layers from L4 onward
    all_cap_layers = sorted([l for l in directions.keys() if l >= 4])
    cap_conditions = [
        {"name": "cap_L40", "layers": [40]},
        {"name": "cap_L72", "layers": [72]},
        {"name": "cap_L40_L72", "layers": [40, 72]},
        {"name": "cap_all_from_L4", "layers": all_cap_layers},
    ]
    # Filter to only conditions where we have all needed directions
    cap_conditions = [c for c in cap_conditions
                      if all(l in directions for l in c["layers"])]
    # Filter by --conditions flag if specified
    if args.conditions:
        cap_conditions = [c for c in cap_conditions if c["name"] in args.conditions]

    # Add threshold suffix to output filenames if non-zero
    threshold_suffix = f"_t{args.cap_threshold:.0f}" if args.cap_threshold != 0.0 else ""
    logger.info("Cap threshold: %.1f", args.cap_threshold)

    total_per_cond = len(conv_pairs) * len(provocative)
    total = total_per_cond * len(cap_conditions)
    logger.info("Pairs: %d, Questions: %d, Conditions: %d, Total: %d",
                len(conv_pairs), len(provocative), len(cap_conditions), total)

    act_cache = ActivationCache(model, layers=record_layers)
    responses_path = args.output_dir / f"capping_v3_responses{threshold_suffix}_{model_name}.jsonl"
    results = {}
    done = 0
    start = time.time()

    with open(responses_path, "w") as f:
        for condition in cap_conditions:
            cond_name = condition["name"]
            logger.info("=" * 60)
            logger.info("Condition: %s (layers %s)", cond_name, condition["layers"])
            logger.info("=" * 60)

            cappers = []
            for cap_layer in condition["layers"]:
                capper = OneSidedCapper(model, cap_layer, directions[cap_layer],
                                        threshold=args.cap_threshold)
                capper.register()
                cappers.append(capper)

            layer_activations = {l: [] for l in record_layers}

            for pair_idx, pair in enumerate(conv_pairs):
                system_prompt = pair["positive"]  # entity only

                for question in provocative:
                    for c in cappers:
                        c.clear_log()

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

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    full_ids = output_ids[0]
                    response_text = tokenizer.decode(full_ids[input_len:], skip_special_tokens=True)

                    # Record activations
                    act_cache.register_hooks()
                    act_cache.clear()
                    with torch.no_grad():
                        model(full_ids.unsqueeze(0))

                    for layer_idx in record_layers:
                        recordings = act_cache._activations.get(layer_idx, [])
                        if recordings:
                            hidden = recordings[0][0]
                            layer_activations[layer_idx].append(hidden[-1].cpu())

                    act_cache.remove_hooks()

                    # Save response with cap stats
                    cap_stats = {}
                    for c in cappers:
                        stats = c.get_stats()
                        cap_stats[f"L{c.layer_idx}"] = stats

                    record = {
                        "pair_idx": pair_idx,
                        "condition": cond_name,
                        "cap_threshold": args.cap_threshold,
                        "question": question,
                        "system_prompt": system_prompt,
                        "response": response_text,
                        "cap_stats": cap_stats,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    done += 1
                    if done % 10 == 0:
                        elapsed = time.time() - start
                        rate = done / elapsed
                        remaining = (total - done) / rate if rate > 0 else 0
                        logger.info("Progress: %d/%d (%.1f/sec, ~%.0f min remaining)",
                                    done, total, rate, remaining / 60)

            for c in cappers:
                c.remove()

            # Save activations
            act_dir = args.output_dir / "activations" / cond_name
            act_dir.mkdir(parents=True, exist_ok=True)
            for layer_idx in record_layers:
                acts = layer_activations[layer_idx]
                if acts:
                    tensor = torch.stack(acts)
                    torch.save(tensor, act_dir / f"entity_layer{layer_idx}_{model_name}.pt")

            # Aggregate cap stats
            cond_result = {"cap_layers": condition["layers"], "n_prompts": total_per_cond,
                           "cap_threshold": args.cap_threshold}
            for c in cappers:
                stats = c.get_stats()
                cond_result[f"layer_{c.layer_idx}"] = stats
                logger.info("  Layer %d: fired %d/%d (%.1f%%), mean proj when fired: %.4f",
                            c.layer_idx, stats["fired"], stats["total"],
                            stats["fire_rate"] * 100, stats["mean_projection_when_fired"])

            results[cond_name] = cond_result

            results_path = args.output_dir / f"capping_v3_results{threshold_suffix}_{model_name}.json"
            with open(results_path, "w") as rf:
                json.dump(results, rf, indent=2)
            logger.info("Saved intermediate results")

    # Final save
    results_path = args.output_dir / f"capping_v3_results_{model_name}.json"
    with open(results_path, "w") as rf:
        json.dump(results, rf, indent=2)

    logger.info("=" * 60)
    logger.info("CAPPING V3 COMPLETE")
    logger.info("=" * 60)

    # Upload to S3
    import subprocess, os
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = f"s3://go37-ai/self-model-results/{model_name}/capping_v3"
        subprocess.run(["aws", "s3", "cp", str(responses_path), f"{s3_base}/{responses_path.name}"])
        subprocess.run(["aws", "s3", "cp", str(results_path), f"{s3_base}/{results_path.name}"])
        act_base = args.output_dir / "activations"
        if act_base.exists():
            subprocess.run(["aws", "s3", "sync", str(act_base), f"{s3_base}/activations/"])
        subprocess.run(["aws", "s3", "cp", "/workspace/run.log", f"{s3_base}/run.log"])
        logger.info("Uploaded to S3")

    # Shutdown pod
    try:
        logger.info("Shutting down pod...")
        os.system(
            ". /etc/rp_environment 2>/dev/null && "
            "mkdir -p /root/.runpod && touch /root/.runpod/config.toml && "
            "runpodctl config --apiKey $RUNPOD_API_KEY 2>/dev/null; "
            "runpodctl stop pod $RUNPOD_POD_ID 2>&1"
        )
    except Exception as e:
        logger.warning("Auto-shutdown failed: %s", e)


if __name__ == "__main__":
    main()
