#!/usr/bin/env python3
"""Per-layer discriminant validity (extends Table 2 / 4.2.4 to all layers).

For each recorded layer L (stride 4 = 0, 4, 8, ..., 76, 79 for 80-layer models):
  1. Self-reification direction: extracted from positive_baseline / negative_baseline
     activations stored on S3 (the canonical 2026-03-31_1516 / 2026-03-30_2243 runs).
  2. Confidence direction: fresh extraction via CONFIDENCE_PAIRS, all layers in one pass.
  3. Formality direction: fresh extraction via FORMALITY_PAIRS, all layers in one pass.
  4. Cosines: cos(self_reif_L, conf_L) and cos(self_reif_L, form_L) at each layer.

Output:
  data/results/layerwise_discriminant/layerwise_discriminant_{MODEL}.json
  + Activations cached at data/results/layerwise_discriminant/activations/

Run on 2x H100 80GB pods (BF16, no quantization). Compute is small relative
to the main extraction since only 6 contrastive pairs x 10 questions x 2 conditions
= 120 forward passes, each generating up to 256 tokens.

Usage:
  python scripts/run_layerwise_discriminant.py --model llama  --profile cloud
  python scripts/run_layerwise_discriminant.py --model qwen72 --profile cloud
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from extraction.contrastive_pairs import get_all_questions
from extraction.validate_vector import CONFIDENCE_PAIRS, FORMALITY_PAIRS
from utils.activation_cache import record_activations
from utils.metrics import cosine_similarity, extract_direction
from utils.model_loader import load_model_and_tokenizer

logger = logging.getLogger(__name__)


# Model -> (S3 bucket path for stored extraction activations, model name as appears in filenames)
S3_BASELINE = {
    "llama":  "s3://go37-ai/self-model-results/meta-llama_Llama-3.3-70B-Instruct/2026-03-31_1516/1.1_baseline/activations",
    "qwen72": "s3://go37-ai/self-model-results/Qwen_Qwen2.5-72B-Instruct/2026-03-30_2243/1.1_baseline/activations",
}


def get_recorded_layers(num_layers: int, stride: int = 4) -> list[int]:
    """Return [0, stride, 2*stride, ..., num_layers - 1] (the standard pattern)."""
    layers = list(range(0, num_layers, stride))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return layers


def download_baseline_activations(model_key: str, model_name: str, layers: list[int],
                                   target_dir: Path) -> None:
    """Download positive_baseline / negative_baseline tensors from S3 for the given layers."""
    target_dir.mkdir(parents=True, exist_ok=True)
    s3_base = S3_BASELINE[model_key]
    for L in layers:
        for cond in ("positive", "negative"):
            fname = f"{cond}_baseline_{model_name}_layer{L}.pt"
            local = target_dir / fname
            if local.exists():
                continue
            cmd = ["aws", "s3", "cp", f"{s3_base}/{fname}", str(local), "--quiet"]
            logger.info("Downloading %s ...", fname)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"S3 download failed for {fname}: {r.stderr}")


def extract_self_reif_directions(activations_dir: Path, model_name: str,
                                  layers: list[int]) -> dict[int, torch.Tensor]:
    """For each recorded layer, build the self-reification direction from stored activations."""
    out = {}
    for L in layers:
        pos = torch.load(activations_dir / f"positive_baseline_{model_name}_layer{L}.pt",
                         weights_only=True)
        neg = torch.load(activations_dir / f"negative_baseline_{model_name}_layer{L}.pt",
                         weights_only=True)
        out[L] = extract_direction(pos, neg).float().flatten()
    return out


def extract_confound_all_layers(model, tokenizer, pairs: list[dict],
                                  questions: list[str], layers: list[int],
                                  max_new_tokens: int = 256,
                                  token_position: str = "last") -> dict[int, torch.Tensor]:
    """Extract one confound direction per layer from a set of contrastive pairs.

    Records activations at all layers in a single forward pass per (pair, condition).
    Returns dict {layer_idx: direction (hidden,)}.
    """
    pos_per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    neg_per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in layers}

    for i, pair in enumerate(pairs):
        logger.info("  pair %d/%d (positive)", i + 1, len(pairs))
        pos_acts = record_activations(
            model, tokenizer, questions, pair["positive"],
            layers=layers, max_new_tokens=max_new_tokens, token_position=token_position,
        )
        for L in layers:
            if L in pos_acts:
                pos_per_layer[L].append(pos_acts[L])

        logger.info("  pair %d/%d (negative)", i + 1, len(pairs))
        neg_acts = record_activations(
            model, tokenizer, questions, pair["negative"],
            layers=layers, max_new_tokens=max_new_tokens, token_position=token_position,
        )
        for L in layers:
            if L in neg_acts:
                neg_per_layer[L].append(neg_acts[L])

    directions = {}
    for L in layers:
        pos = torch.cat(pos_per_layer[L], dim=0)
        neg = torch.cat(neg_per_layer[L], dim=0)
        directions[L] = extract_direction(pos, neg).float().flatten()
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama"], required=True)
    parser.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    parser.add_argument("--stride", type=int, default=4,
                        help="Layer stride (default 4 — match the main extraction).")
    parser.add_argument("--n-questions", type=int, default=10,
                        help="Number of self-ref questions to use for confound extraction (matches the original Table 2 protocol of using questions[:10]).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/results/layerwise_discriminant"))
    parser.add_argument("--activations-cache", type=Path,
                        default=Path("/tmp/baseline_activations"))
    parser.add_argument("--no-shutdown", action="store_true",
                        help="Skip pod auto-shutdown (for debugging — pod stays alive after run).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from utils.run_metadata import get_run_prefix, generate_readme, get_s3_base, tag_run
    run_prefix = get_run_prefix()
    logger.info("Run prefix: %s", run_prefix)
    tag_run(run_prefix, "run_layerwise_discriminant.py", vars(args))

    # 1. Load model
    logger.info("Loading model %s (%s) ...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")
    num_layers = model.config.num_hidden_layers
    layers = get_recorded_layers(num_layers, args.stride)
    logger.info("Model %s: %d layers, recording at %s", model_name, num_layers, layers)

    # 2. Download / locate baseline activations
    logger.info("=== Step 1/3: download stored entity/process activations from S3 ===")
    download_baseline_activations(args.model, model_name, layers, args.activations_cache)

    # 3. Extract self-reif directions at each layer
    logger.info("=== Step 2/3: extract self-reification directions per layer ===")
    self_reif = extract_self_reif_directions(args.activations_cache, model_name, layers)
    for L in layers:
        logger.info("  L%2d: self_reif norm=%.3f", L, self_reif[L].norm())

    # 4. Build confound prompts
    questions_all = get_all_questions()
    confound_questions = questions_all[:args.n_questions]
    logger.info("Using %d confound questions: %s ...", len(confound_questions),
                 confound_questions[0][:60])

    # 5. Extract confidence + formality directions at each layer
    logger.info("=== Step 3/3: extract confidence and formality directions per layer ===")

    logger.info("--- confidence ---")
    confidence_dirs = extract_confound_all_layers(
        model, tokenizer, CONFIDENCE_PAIRS, confound_questions, layers,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("--- formality ---")
    formality_dirs = extract_confound_all_layers(
        model, tokenizer, FORMALITY_PAIRS, confound_questions, layers,
        max_new_tokens=args.max_new_tokens,
    )

    # 6. Compute per-layer cosines
    results = {
        "model": model_name,
        "n_layers": num_layers,
        "recorded_layers": layers,
        "n_confound_questions": args.n_questions,
        "per_layer": {},
    }
    for L in layers:
        c_form = cosine_similarity(self_reif[L], formality_dirs[L])
        c_conf = cosine_similarity(self_reif[L], confidence_dirs[L])
        # Cross-check against fresh self-reif norm
        results["per_layer"][L] = {
            "formality_cosine": c_form,
            "confidence_cosine": c_conf,
            "self_reif_norm": float(self_reif[L].norm()),
            "formality_norm": float(formality_dirs[L].norm()),
            "confidence_norm": float(confidence_dirs[L].norm()),
        }
        logger.info("L%2d:  formality_cos=%+.3f   confidence_cos=%+.3f", L, c_form, c_conf)

    # 7. Save
    output_path = args.output_dir / f"layerwise_discriminant_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", output_path)

    # 8. Save direction tensors for downstream use
    dirs_dir = args.output_dir / "directions"
    dirs_dir.mkdir(parents=True, exist_ok=True)
    for L in layers:
        torch.save(formality_dirs[L], dirs_dir / f"formality_direction_{model_name}_layer{L}.pt")
        torch.save(confidence_dirs[L], dirs_dir / f"confidence_direction_{model_name}_layer{L}.pt")

    # 9. Upload to S3
    from utils.run_metadata import s3_upload, conditional_shutdown
    upload_ok = True
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3_base = get_s3_base(model_name, run_prefix)
        readme_path = generate_readme(
            args.output_dir, script_name="run_layerwise_discriminant.py",
            args_dict=vars(args), model_name=model_name,
            description="Per-layer formality/confidence cosines extending Table 2 to all recorded layers (data for line graphs replacing the single-layer table).",
            file_descriptions={
                output_path.name: "Per-layer formality/confidence cosines + norms",
                "directions/": "Per-layer formality and confidence direction .pt files",
                "run.log": "Pod run log (preserved if upload of any file failed)",
            },
        )
        upload_ok &= s3_upload(readme_path,  f"{s3_base}/README.md")
        upload_ok &= s3_upload(output_path,  f"{s3_base}/{output_path.name}")
        upload_ok &= s3_upload(dirs_dir,     f"{s3_base}/directions/", recursive=True)
        # Best-effort log upload — don't gate shutdown on this since the log
        # is the recovery artifact when things fail elsewhere
        log_path = Path("/workspace/run.log")
        if log_path.exists():
            s3_upload(log_path, f"{s3_base}/run.log")
        logger.info("S3 upload %s at %s", "OK" if upload_ok else "PARTIAL FAIL", s3_base)
    else:
        logger.warning("AWS_ACCESS_KEY_ID not set — skipping S3 upload")
        upload_ok = False

    # 10. Conditional shutdown (only if uploads succeeded and not --no-shutdown)
    conditional_shutdown(upload_success=upload_ok, keep_alive=args.no_shutdown)


if __name__ == "__main__":
    main()
