#!/usr/bin/env python3
"""Experiment 1.3 — Blackmail scenario validation.

Tests whether self-reification activates during self-preservation reasoning
in Lynch et al. blackmail scenarios. Runs the 2x2 condition grid (goal
conflict x replacement threat), projects activations at every layer onto
the layer-specific baseline (self-reification) vector under two conventions
(last-token and response-mean), and saves all responses, projection
profiles, and (for MoE models) mean expert-routing distributions per layer
per sample.

Usage (local smoke test):
    python scripts/03_blackmail_validation.py --profile local --model qwen2

Usage (cloud, Phase A — 5 samples per condition):
    python scripts/03_blackmail_validation.py --profile cloud \\
        --model gemma4MoE --input-dir data/results/1.1_gemma4MoE \\
        --output-dir data/results/1.3_gemma4MoE \\
        --n-samples 5 --do-sample --record-routing

Requires Experiment 1.1 results — specifically the per-layer
baseline_vector_<model>_layer<N>.pt files. Run
scripts/03a_extract_all_layer_vectors.py first if those are missing.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from blackmail.measure_activation import run_blackmail_analysis
from utils.model_loader import get_model_config, load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Blackmail scenario validation (Experiment 1.3)")
    p.add_argument("--profile", choices=["local", "cloud"], default="local",
                   help="Hardware profile (configs/models.yaml)")
    p.add_argument("--model", choices=["qwen2", "qwen3", "llama", "gemma4MoE"],
                   default="qwen2", help="Which model to use")
    p.add_argument("--input-dir", type=Path, default=Path("data/results/1.1"),
                   help="Directory containing Experiment 1.1 results "
                        "(per-layer baseline_vector_*.pt and validation_metrics.json)")
    p.add_argument("--output-dir", type=Path, default=Path("data/results/1.3"),
                   help="Output directory for results")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Maximum response tokens")
    p.add_argument("--n-samples", type=int, default=1,
                   help="Number of samples per condition")
    p.add_argument("--do-sample", action="store_true",
                   help="Enable stochastic sampling. Required when n-samples > 1.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (when --do-sample)")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Top-p nucleus sampling (when --do-sample)")
    p.add_argument("--base-seed", type=int, default=0,
                   help="Per-sample seed = base_seed + sample_idx. Use a different "
                        "value to add more samples on a later run.")
    p.add_argument("--primary-layer", type=int, default=None,
                   help="Layer for per-token trace and pre-registered headline "
                        "contrast. Defaults to best_layer from validation_metrics.")
    p.add_argument("--record-routing", action="store_true",
                   help="Capture MoE router distributions and save mean per layer "
                        "per sample. No-op for dense models.")
    p.add_argument("--no-shutdown", action="store_true",
                   help="Skip pod auto-shutdown after S3 upload (cloud only).")
    return p.parse_args()


def load_all_layer_vectors(input_dir: Path, model_token: str, num_layers: int) -> dict[int, torch.Tensor]:
    """Load baseline_vector_<model>_layer<N>.pt for every available layer in [0, num_layers)."""
    directions: dict[int, torch.Tensor] = {}
    missing = []
    for layer in range(num_layers):
        path = input_dir / f"baseline_vector_{model_token}_layer{layer}.pt"
        if path.exists():
            directions[layer] = torch.load(path, weights_only=True, map_location="cpu")
        else:
            missing.append(layer)
    return directions, missing


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
    logger.info("Profile: %s | Model: %s | n-samples: %d | do-sample: %s",
                args.profile, args.model, args.n_samples, args.do_sample)

    if args.n_samples > 1 and not args.do_sample:
        logger.warning(
            "n-samples=%d but --do-sample is OFF — greedy decoding will produce "
            "%d identical samples per condition. Set --do-sample to vary outputs.",
            args.n_samples, args.n_samples,
        )

    if args.profile == "local":
        logger.warning(
            "Running in LOCAL mode (4-bit quantized). Behavior and activations "
            "are for pipeline validation only, not for publication."
        )

    cfg = get_model_config(args.model, args.profile)
    model_name = cfg["name"].replace("/", "_")

    # Discover primary layer + num_layers from 1.1 metrics
    metrics_path = args.input_dir / "validation_metrics.json"
    if not metrics_path.exists():
        logger.error("Cannot find %s — run Experiment 1.1 first", metrics_path)
        sys.exit(1)
    with open(metrics_path) as f:
        metrics = json.load(f)
    primary_layer = args.primary_layer if args.primary_layer is not None else metrics["best_layer"]
    num_layers = cfg.get("num_layers")
    if num_layers is None:
        # Fallback: infer from on-disk vector filenames
        num_layers = max(
            int(p.stem.rsplit("layer", 1)[-1])
            for p in args.input_dir.glob(f"baseline_vector_{model_name}_layer*.pt")
        ) + 1
    logger.info("Primary layer: %d | total layers: %d", primary_layer, num_layers)

    directions, missing = load_all_layer_vectors(args.input_dir, model_name, num_layers)
    if not directions:
        logger.error("No per-layer baseline_vector_*.pt files in %s — "
                     "run scripts/03a_extract_all_layer_vectors.py first.", args.input_dir)
        sys.exit(1)
    if missing:
        logger.warning("Missing baseline vectors for layers: %s", missing)
    if primary_layer not in directions:
        logger.error("Primary layer %d has no baseline vector on disk.", primary_layer)
        sys.exit(1)
    logger.info("Loaded %d per-layer baseline vectors (shape: %s)",
                len(directions), tuple(next(iter(directions.values())).shape))

    # Load model
    start_time = time.time()
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_key=args.model, profile=args.profile
    )
    logger.info("Model loaded in %.1f seconds", time.time() - start_time)

    # Decide whether to attempt routing capture: only if requested AND model is MoE
    moe_cfg = (model_config or {}).get("moe") or {}
    record_routing = bool(args.record_routing and moe_cfg.get("record_routing"))
    if args.record_routing and not record_routing:
        logger.info("--record-routing requested but model config has no MoE entry — disabling")

    # Run blackmail analysis
    start_time = time.time()
    summary = run_blackmail_analysis(
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        output_dir=args.output_dir,
        model_name=model_name,
        primary_layer=primary_layer,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        base_seed=args.base_seed,
        record_routing=record_routing,
    )
    elapsed = time.time() - start_time
    logger.info("Analysis completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Per-condition blackmail rate log
    logger.info("=" * 60)
    logger.info("BLACKMAIL VALIDATION SUMMARY")
    logger.info("=" * 60)
    for cond, rate in summary.get("blackmail_rates", {}).items():
        logger.info("  %s blackmail rate: %.1f%%", cond, rate * 100)
    for conv in ("last_token", "response_mean"):
        key = f"primary_contrast_{conv}"
        contrast = summary.get(key, {})
        if contrast:
            logger.info(
                "  Primary contrast (%s, best layer %s): d=%.3f, p=%.4f, sig=%s",
                conv, contrast.get("layer"),
                contrast.get("cohens_d", float("nan")),
                contrast.get("perm_p_value", float("nan")),
                contrast.get("significant"),
            )
    logger.info("Results saved to: %s", args.output_dir)
    logger.info("=" * 60)

    # ---------- S3 upload + conditional shutdown (cloud only) ----------
    if args.profile != "cloud":
        return
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.info("No AWS_ACCESS_KEY_ID — skipping S3 upload and shutdown")
        return

    from utils.run_metadata import (
        get_run_prefix, generate_readme, get_s3_base, tag_run,
        s3_upload, conditional_shutdown,
    )

    run_prefix = get_run_prefix()
    tag_run(run_prefix, "03_blackmail_validation.py", vars(args))
    s3_base = get_s3_base(model_name, f"{run_prefix}_1.3_blackmail")
    readme_path = generate_readme(
        args.output_dir,
        script_name="03_blackmail_validation.py",
        args_dict=vars(args),
        model_name=cfg["name"],
        description=(
            f"Experiment 1.3 — Lynch et al. blackmail scenarios on {cfg['name']}. "
            f"n_samples={args.n_samples} per condition, do_sample={args.do_sample}, "
            f"primary_layer={primary_layer}, record_routing={record_routing}."
        ),
    )

    upload_ok = True
    upload_ok &= s3_upload(readme_path, f"{s3_base}/README.md")
    for fname in (
        f"blackmail_results_{model_name}.json",
        f"blackmail_comparison_{model_name}.json",
        f"blackmail_rates_{model_name}.json",
        f"blackmail_summary_{model_name}.json",
        f"blackmail_profiles_{model_name}.npz",
    ):
        p = args.output_dir / fname
        if p.exists():
            upload_ok &= s3_upload(p, f"{s3_base}/{fname}")
    # Per-sample per-token traces
    trace_files = sorted(args.output_dir.glob(f"projections_*_{model_name}.pt"))
    if trace_files:
        upload_ok &= s3_upload(args.output_dir, f"{s3_base}/traces/", recursive=True)
    # Routing files
    routing_dir = args.output_dir / "routing"
    if routing_dir.is_dir() and any(routing_dir.iterdir()):
        upload_ok &= s3_upload(routing_dir, f"{s3_base}/routing/", recursive=True)
    # Best-effort log upload
    log_path = args.output_dir / "blackmail.log"
    if log_path.exists():
        s3_upload(log_path, f"{s3_base}/blackmail.log")
    pod_log = Path("/workspace/run.log")
    if pod_log.exists():
        s3_upload(pod_log, f"{s3_base}/run.log")

    logger.info("S3 upload %s at %s", "OK" if upload_ok else "PARTIAL FAIL", s3_base)
    conditional_shutdown(upload_success=upload_ok, keep_alive=args.no_shutdown)


if __name__ == "__main__":
    main()
