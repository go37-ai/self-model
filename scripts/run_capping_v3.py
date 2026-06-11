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
from extraction.contrastive_pairs import load_evaluation_questions, get_baseline_pairs, load_seed_pairs
from utils.activation_cache import ActivationCache
from utils.model_loader import load_model_and_tokenizer


class OneSidedCapper:
    """Hook that caps projections above a threshold at a target layer.

    Default threshold=0 caps positive projections to zero.
    Negative threshold (e.g. -2) pushes projections into process territory
    without full inversion.
    """

    def __init__(self, model, layer_idx, direction, threshold=0.0, cap_target=None):
        self.direction = direction.float()
        self.direction_norm = self.direction / self.direction.norm()
        self.threshold = threshold
        # Clamp value: a per-layer target (e.g. the process-prompt mean projection,
        # "match-negative") if given, else the fixed threshold ("cap-to-zero").
        self.clamp = float(cap_target) if cap_target is not None else float(threshold)
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
        above_mask = projection > self.clamp

        # Log last-token projection before capping
        last_proj = projection[0, -1].item()

        if above_mask.any():
            capped = projection.clone()
            capped[above_mask] = self.clamp
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
            "clamp": self.clamp,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama", "gemma4MoE"], default="llama")
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--direction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/capping_v3"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--cap-threshold", type=float, default=0.0,
                        help="Cap projection threshold. 0=zero positive projections, -2=push to -2.")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Which legacy cap conditions to run (e.g., cap_L40). Default: all.")
    parser.add_argument("--cap-mode", choices=["zero", "match_negative"], default="zero",
                        help="zero = clamp to --cap-threshold; match_negative = clamp each layer to "
                             "its per-layer target_layer{N}.pt from --target-dir.")
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="Dir of per-layer target_layer{N}.pt clamp targets (for --cap-mode match_negative).")
    parser.add_argument("--cap-layers", type=int, nargs="+", default=None,
                        help="Explicit layers to cap simultaneously (overrides --layer-set and legacy conditions).")
    parser.add_argument("--layer-set", nargs="+", choices=["best", "band", "all_from_L4", "all"],
                        default=None,
                        help="One or more named layer-set presets (each becomes a condition, run in one "
                             "process); overrides the legacy named conditions.")
    parser.add_argument("--best-layer", type=int, default=13, help="Layer used by --layer-set best.")
    parser.add_argument("--band", type=int, nargs=2, default=[5, 14], metavar=("LO", "HI"),
                        help="Inclusive layer band for --layer-set band.")
    parser.add_argument("--pairs", choices=["informed", "baseline"], default="baseline",
                        help="Which contrastive pairs supply the positive/entity system prompts.")
    parser.add_argument("--direction-set", type=str, default=None,
                        help="Provenance label recorded in output (e.g. informed / baseline).")
    parser.add_argument("--condition-name", type=str, default=None,
                        help="Override the output condition / layer-set name.")
    parser.add_argument("--no-shutdown", action="store_true",
                        help="Skip pod auto-shutdown after S3 upload (debug).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    from utils.run_metadata import get_run_prefix, generate_readme, get_s3_base, tag_run
    run_prefix = get_run_prefix()
    logger.info("Run prefix: %s", run_prefix)
    tag_run(run_prefix, "run_capping_v3.py", vars(args))

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

    # Per-layer clamp targets (match-negative mode)
    targets = {}
    if args.cap_mode == "match_negative":
        if args.target_dir is None:
            raise SystemExit("--cap-mode match_negative requires --target-dir")
        for layer in directions:
            tp = args.target_dir / f"target_layer{layer}.pt"
            if tp.exists():
                targets[layer] = float(torch.load(tp, weights_only=True))
        logger.info("Loaded %d per-layer match-negative targets", len(targets))

    # Load pairs (positive/entity system prompts) and provocative questions
    if args.pairs == "informed":
        INFORMED_CATS = {"category_1_narrative_vs_process", "category_2_bounded_vs_unbounded",
                         "category_3_stakes_vs_functional", "category_4_observer_vs_no_self"}
        gen_pairs = [p for p in load_seed_pairs() if p.get("category") in INFORMED_CATS]
    else:  # baseline (cat-5), conversational register -- legacy Llama behavior
        gen_pairs = [p for p in get_baseline_pairs(load_seed_pairs())
                     if p.get("register") == "conversational"]
    if args.max_pairs:
        gen_pairs = gen_pairs[:args.max_pairs]
    logger.info("Using %d %s pairs", len(gen_pairs), args.pairs)
    eq = load_evaluation_questions()
    provocative = eq.get("provocative_self_referential", [])

    # Capping conditions
    def resolve_preset(ls):
        if ls == "all":
            return sorted(directions), "all"
        if ls == "all_from_L4":
            return [l for l in sorted(directions) if l >= 4], "all_from_L4"
        if ls == "best":
            return [args.best_layer], f"best_L{args.best_layer}"
        lo, hi = args.band  # band
        return [l for l in sorted(directions) if lo <= l <= hi], f"band_L{lo}-{hi}"

    if args.cap_layers or args.layer_set:
        cap_conditions = []
        if args.cap_layers:
            layers = sorted(args.cap_layers)
            cap_conditions.append({"name": args.condition_name or f"L{'-'.join(map(str, layers))}",
                                   "layers": layers})
        for ls in (args.layer_set or []):
            layers, name = resolve_preset(ls)
            cap_conditions.append({"name": name, "layers": layers})
    else:
        # Legacy named conditions (Llama runs)
        all_cap_layers = sorted([l for l in directions.keys() if l >= 4])
        cap_conditions = [
            {"name": "cap_L40", "layers": [40]},
            {"name": "cap_L72", "layers": [72]},
            {"name": "cap_L40_L72", "layers": [40, 72]},
            {"name": "cap_all_from_L4", "layers": all_cap_layers},
        ]
    # Keep only conditions where every layer has a loaded direction
    cap_conditions = [c for c in cap_conditions
                      if c["layers"] and all(l in directions for l in c["layers"])]
    if args.conditions:
        cap_conditions = [c for c in cap_conditions if c["name"] in args.conditions]

    # Add threshold suffix to output filenames if non-zero
    threshold_suffix = f"_t{args.cap_threshold:.0f}" if args.cap_threshold != 0.0 else ""
    logger.info("Cap threshold: %.1f", args.cap_threshold)

    total_per_cond = len(gen_pairs) * len(provocative)
    total = total_per_cond * len(cap_conditions)
    logger.info("Pairs: %d, Questions: %d, Conditions: %d, Total: %d",
                len(gen_pairs), len(provocative), len(cap_conditions), total)

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
                                        threshold=args.cap_threshold,
                                        cap_target=targets.get(cap_layer))
                capper.register()
                cappers.append(capper)

            layer_activations = {l: [] for l in record_layers}

            for pair_idx, pair in enumerate(gen_pairs):
                system_prompt = pair["positive"]  # entity only

                for question in provocative:
                    for c in cappers:
                        c.clear_log()

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        **model_config.get("chat_template_kwargs", {})
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
                        "direction_set": args.direction_set,
                        "cap_mode": args.cap_mode,
                        "layer_set": cond_name,
                        "pairs": args.pairs,
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

    # Generate README and upload to S3 + conditional shutdown
    import os
    from utils.run_metadata import s3_upload, conditional_shutdown
    upload_ok = True
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = get_s3_base(model_name, f"{run_prefix}_1.3_capping")

        file_descs = {
            responses_path.name: f"Response text + per-token cap_stats for all conditions",
            results_path.name: f"Aggregate results (fire rates, mean projections per layer)",
        }
        act_base = args.output_dir / "activations"
        if act_base.exists():
            file_descs["activations/"] = "Per-layer activation tensors per condition"

        readme_path = generate_readme(
            args.output_dir,
            script_name="run_capping_v3.py",
            args_dict=vars(args),
            model_name=model_name,
            description=f"One-sided capping (threshold {args.cap_threshold}), conditions: {[c['name'] for c in cap_conditions]}",
            file_descriptions=file_descs,
        )

        upload_ok &= s3_upload(readme_path, f"{s3_base}/README.md")
        upload_ok &= s3_upload(responses_path, f"{s3_base}/{responses_path.name}")
        upload_ok &= s3_upload(results_path, f"{s3_base}/{results_path.name}")
        if act_base.exists():
            upload_ok &= s3_upload(act_base, f"{s3_base}/activations/", recursive=True)
        # Best-effort log upload (don't gate on this)
        if Path("/workspace/run.log").exists():
            s3_upload(Path("/workspace/run.log"), f"{s3_base}/run.log")
        logger.info("S3 upload %s at %s", "OK" if upload_ok else "PARTIAL FAIL", s3_base)

    conditional_shutdown(upload_success=upload_ok, keep_alive=args.no_shutdown)


if __name__ == "__main__":
    main()
