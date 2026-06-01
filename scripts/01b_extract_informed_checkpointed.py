#!/usr/bin/env python3
"""Experiment 1.1 — Informed (categories 1-4) extraction with per-category checkpointing.

A checkpointed variant of `01_extract_vector.py --pairs informed`. The stock
pipeline collects activations for ALL 20 informed pairs in memory and only
saves once at the very end, so a GPU/pod failure loses the whole run and no
preliminary results are visible until completion.

This script instead processes the four informed categories one at a time
(5 pairs each). After EACH category it:
  1. Saves that category's positive/negative activations, response texts, and
     a manifest to disk.
  2. Uploads them to S3 immediately.
  3. Computes and saves that category's split-half reliability so a partial
     result is visible ~1/4 of the way through the run.

On `--resume` it pulls any already-uploaded category checkpoints from S3 and
skips collecting them, so a fresh pod can pick up where a dead one left off.
After all four categories are done it concatenates them into the standard
`positive_informed_{model}` / `negative_informed_{model}` activation set and
produces the same combined-direction, per-category-vector, category-similarity,
and reliability outputs as the stock pipeline, so downstream analysis scripts
work unchanged.

Pod shutdown follows the recorded best practice (reference_pod_setup.md):
uploads gate shutdown via `conditional_shutdown`, and `--no-shutdown` forces
the pod to stay alive for debugging.

Usage (first launch — note the run prefix it prints):
    python scripts/01b_extract_informed_checkpointed.py --model llama --profile cloud

Resume on a fresh pod after a failure (reuse the SAME prefix it printed):
    python scripts/01b_extract_informed_checkpointed.py --model llama --profile cloud \
        --resume --run-id 2026-06-01_1530
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from extraction.contrastive_pairs import (
    INFORMED_CATEGORIES,
    get_all_questions,
    get_informed_pairs,
    get_pairs_by_category,
    load_evaluation_questions,
    load_seed_pairs,
)
from extraction.extract_vector import (
    collect_condition_activations,
    extract_all_directions,
    select_best_layer,
)
from utils.activation_cache import load_activations, save_activations, save_manifest
from utils.metrics import (
    extract_direction,
    pairwise_cosine_matrix,
    split_half_reliability,
)
from utils.model_loader import load_model_and_tokenizer
from utils.run_metadata import (
    conditional_shutdown,
    generate_readme,
    get_run_prefix,
    get_s3_base,
    s3_download,
    s3_upload,
    tag_run,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Checkpointed informed (cat 1-4) extraction (Experiment 1.1)"
    )
    p.add_argument("--model", choices=["qwen2", "qwen3", "qwen72", "llama", "gemma4MoE"],
                   default="llama")
    p.add_argument("--profile", choices=["local", "cloud"], default="cloud")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/results/1.1_informed_checkpointed"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--token-position", choices=["last", "mean"], default="last")
    p.add_argument("--n-splits", type=int, default=100,
                   help="Random splits for split-half reliability.")
    p.add_argument("--resume", action="store_true",
                   help="Pull completed category checkpoints from S3 and skip them.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Stable run prefix (YYYY-MM-DD_HHMM). Pass the SAME value to "
                        "resume into the same S3 location. Auto-generated if omitted.")
    p.add_argument("--no-shutdown", action="store_true",
                   help="Skip pod auto-shutdown after S3 upload (debug).")
    return p.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "extraction.log"),
        ],
    )

    # Stable run prefix so a resumed run reads/writes the SAME S3 location.
    run_prefix = args.run_id or get_run_prefix()
    run_label = f"{run_prefix}_1.1_informed"
    logger.info("=" * 60)
    logger.info("Experiment 1.1 — Informed extraction (checkpointed, per-category)")
    logger.info("=" * 60)
    logger.info("Run prefix: %s   (resume with: --resume --run-id %s)",
                run_prefix, run_prefix)
    tag_run(run_prefix, "01b_extract_informed_checkpointed.py", vars(args))

    if args.profile == "local":
        logger.warning("LOCAL/quantized mode — results are for DEBUGGING ONLY.")

    # --- Load model ---
    t0 = time.time()
    logger.info("Loading model %s (%s)...", args.model, args.profile)
    model, tokenizer, model_config = load_model_and_tokenizer(args.model, args.profile)
    model_name = model_config["name"].replace("/", "_")
    logger.info("Model loaded in %.1f s", time.time() - t0)

    # --- Recording layers: stride from config, always include layer 0 and last ---
    num_layers = model_config["num_layers"]
    layer_stride = model_config.get("layer_stride", 1)
    layers = list(range(0, num_layers, layer_stride))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    logger.info("Recording %d of %d layers (stride %d): %s",
                len(layers), num_layers, layer_stride, layers)

    # MoE routing capture (no-op for dense models like Llama).
    moe_cfg = model_config.get("moe") or {}
    record_routing = bool(moe_cfg.get("record_routing"))
    template_kwargs = model_config.get("chat_template_kwargs") or None

    # --- Pairs and questions ---
    all_pairs = load_seed_pairs()
    informed_pairs = get_informed_pairs(all_pairs)
    pairs_by_cat = get_pairs_by_category(informed_pairs)
    questions = get_all_questions()

    _eq = load_evaluation_questions()
    question_types = (
        ["self_referential"] * len(_eq["self_referential"])
        + ["provocative_self_referential"] * len(_eq.get("provocative_self_referential", []))
        + ["non_self_referential"] * len(_eq["non_self_referential"])
    )
    n_questions = len(questions)
    logger.info("Informed pairs: %d across %d categories | %d questions",
                len(informed_pairs), len(pairs_by_cat), n_questions)

    # --- S3 setup + (optional) resume pull ---
    have_s3 = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
    s3_base = get_s3_base(model_name, run_label) if have_s3 else None
    act_dir = args.output_dir / "activations"
    texts_dir = args.output_dir / "response_texts"
    act_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and have_s3:
        logger.info("Resume: pulling any prior checkpoints from %s ...", s3_base)
        # Best-effort: pull whatever activations/texts already exist on S3 so the
        # per-category "files present locally" check below can skip them.
        s3_download(f"{s3_base}/activations/", act_dir, recursive=True)
        s3_download(f"{s3_base}/response_texts/", texts_dir, recursive=True)

    upload_ok = True  # gates pod shutdown — any required upload failure keeps pod alive

    def cat_prefixes(cat_key):
        return (f"positive_informed_{cat_key}_{model_name}",
                f"negative_informed_{cat_key}_{model_name}")

    def upload_file(local, remote_subpath):
        nonlocal upload_ok
        if have_s3:
            upload_ok &= s3_upload(local, f"{s3_base}/{remote_subpath}")

    # --- Per-category collection with checkpointing ---
    cat_pos: dict[str, dict[int, torch.Tensor]] = {}
    cat_neg: dict[str, dict[int, torch.Tensor]] = {}
    per_cat_reliability: dict[str, dict[int, float]] = {}

    for cat_idx, cat_key in enumerate(INFORMED_CATEGORIES):
        cat_pairs = pairs_by_cat.get(cat_key, [])
        if not cat_pairs:
            logger.warning("No pairs for category %s — skipping", cat_key)
            continue
        pos_prefix, neg_prefix = cat_prefixes(cat_key)

        # Resume check: are this category's activations already complete on disk?
        pos_cached = load_activations(act_dir, pos_prefix, layers)
        neg_cached = load_activations(act_dir, neg_prefix, layers)
        if (args.resume and len(pos_cached) == len(layers)
                and len(neg_cached) == len(layers)):
            logger.info("[cat %d/%d] %s — already complete, skipping collection",
                        cat_idx + 1, len(INFORMED_CATEGORIES), cat_key)
            cat_pos[cat_key] = pos_cached
            cat_neg[cat_key] = neg_cached
        else:
            logger.info("=" * 60)
            logger.info("[cat %d/%d] Collecting %s (%d pairs)",
                        cat_idx + 1, len(INFORMED_CATEGORIES), cat_key, len(cat_pairs))
            logger.info("=" * 60)
            t_cat = time.time()
            (
                pos, neg, _pb, pos_texts, neg_texts, _pr, _nr, _mf
            ) = collect_condition_activations(
                model, tokenizer, cat_pairs, questions, layers,
                max_new_tokens=args.max_new_tokens, token_position=args.token_position,
                record_routing=record_routing, pairs_label=f"informed_{cat_key}",
                question_types=question_types, template_kwargs=template_kwargs,
            )
            cat_pos[cat_key] = pos
            cat_neg[cat_key] = neg

            # Save this category's activations + texts to disk.
            save_activations(pos, act_dir, pos_prefix)
            save_activations(neg, act_dir, neg_prefix)
            with open(texts_dir / f"{pos_prefix}.json", "w") as f:
                json.dump(pos_texts, f)
            with open(texts_dir / f"{neg_prefix}.json", "w") as f:
                json.dump(neg_texts, f)

            elapsed = time.time() - t_cat
            done = cat_idx + 1
            eta_min = elapsed * (len(INFORMED_CATEGORIES) - done) / 60
            logger.info("[cat %d/%d] collected in %.1f min (~%.0f min for remaining %d)",
                        done, len(INFORMED_CATEGORIES), elapsed / 60, eta_min,
                        len(INFORMED_CATEGORIES) - done)

            # Upload this category's checkpoint to S3 immediately.
            for l in layers:
                upload_file(act_dir / f"{pos_prefix}_layer{l}.pt", f"activations/{pos_prefix}_layer{l}.pt")
                upload_file(act_dir / f"{neg_prefix}_layer{l}.pt", f"activations/{neg_prefix}_layer{l}.pt")
            upload_file(texts_dir / f"{pos_prefix}.json", f"response_texts/{pos_prefix}.json")
            upload_file(texts_dir / f"{neg_prefix}.json", f"response_texts/{neg_prefix}.json")

        # Preliminary per-category reliability (visible after each category).
        cat_rel = {}
        for l in layers:
            if l in cat_pos[cat_key] and l in cat_neg[cat_key]:
                cat_rel[l] = split_half_reliability(
                    cat_pos[cat_key][l], cat_neg[cat_key][l], n_splits=args.n_splits
                )
        per_cat_reliability[cat_key] = cat_rel
        if cat_rel:
            best_l = max(cat_rel, key=cat_rel.get)
            logger.info("[cat %d/%d] %s preliminary best layer %d (r=%.4f)",
                        cat_idx + 1, len(INFORMED_CATEGORIES), cat_key, best_l, cat_rel[best_l])
        rel_path = args.output_dir / f"per_category_reliability_{cat_key}_{model_name}.json"
        with open(rel_path, "w") as f:
            json.dump({str(l): v for l, v in cat_rel.items()}, f, indent=2)
        upload_file(rel_path, rel_path.name)
        # Best-effort progress log upload (don't gate shutdown on it).
        if have_s3 and Path("/workspace/run.log").exists():
            s3_upload(Path("/workspace/run.log"), f"{s3_base}/run.log")

    # --- All categories done: build the standard combined informed outputs ---
    logger.info("=" * 60)
    logger.info("All categories collected — assembling combined informed direction")
    logger.info("=" * 60)

    # Concatenate categories in canonical order into the full informed set.
    pos_full = {l: torch.cat([cat_pos[c][l] for c in INFORMED_CATEGORIES if c in cat_pos], dim=0)
                for l in layers}
    neg_full = {l: torch.cat([cat_neg[c][l] for c in INFORMED_CATEGORIES if c in cat_neg], dim=0)
                for l in layers}

    # Save with the standard prefix so downstream analysis scripts find them.
    save_activations(pos_full, act_dir, f"positive_informed_{model_name}")
    save_activations(neg_full, act_dir, f"negative_informed_{model_name}")

    # Concatenate response texts (canonical order) under the standard names.
    def _concat_texts(sign):
        out = []
        for c in INFORMED_CATEGORIES:
            prefix = f"{sign}_informed_{c}_{model_name}"
            fp = texts_dir / f"{prefix}.json"
            if fp.exists():
                out.extend(json.load(open(fp)))
        return out
    pos_texts_all = _concat_texts("positive")
    neg_texts_all = _concat_texts("negative")
    with open(texts_dir / f"positive_informed_{model_name}.json", "w") as f:
        json.dump(pos_texts_all, f)
    with open(texts_dir / f"negative_informed_{model_name}.json", "w") as f:
        json.dump(neg_texts_all, f)

    # Build the combined manifest with GLOBAL pair indices (0..19), matching the
    # stock pipeline's row ordering: all positive rows then all negative rows.
    manifest_rows = []
    for condition in ("positive", "negative"):
        row_idx = 0
        for pair_idx, pair in enumerate(informed_pairs):
            pair_id = f"{pair['category']}_pair_{pair_idx:03d}"
            register = pair.get("register", "untagged")
            for q_idx in range(n_questions):
                manifest_rows.append({
                    "row_idx": row_idx, "condition": condition,
                    "pairs_label": "informed", "pair_id": pair_id,
                    "category": pair["category"], "register": register,
                    "question_id": f"q{q_idx:03d}", "question_type": question_types[q_idx],
                })
                row_idx += 1
    save_manifest(manifest_rows, act_dir, filename=f"manifest_informed_{model_name}.jsonl")

    # Combined direction + best layer by split-half reliability over the full set.
    combined_directions = extract_all_directions(pos_full, neg_full, layers)
    best_layer, reliabilities = select_best_layer(pos_full, neg_full, layers, n_splits=args.n_splits)
    combined_dir = combined_directions[best_layer]
    torch.save(combined_dir,
               args.output_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt")
    with open(args.output_dir / f"layer_reliability_{model_name}.json", "w") as f:
        json.dump({"reliabilities": {str(k): v for k, v in reliabilities.items()},
                   "best_layer": best_layer}, f, indent=2)

    # Per-category vectors at the combined best layer + similarity matrix.
    per_category_vectors = {}
    for c in INFORMED_CATEGORIES:
        if c in cat_pos and best_layer in cat_pos[c] and best_layer in cat_neg[c]:
            per_category_vectors[c] = extract_direction(cat_pos[c][best_layer], cat_neg[c][best_layer])
    torch.save(per_category_vectors,
               args.output_dir / f"per_category_vectors_{model_name}_layer{best_layer}.pt")
    if len(per_category_vectors) > 1:
        cat_sim = pairwise_cosine_matrix(per_category_vectors)
        with open(args.output_dir / f"category_similarity_matrix_{model_name}.json", "w") as f:
            json.dump(cat_sim, f, indent=2)
        logger.info("Category similarity matrix: %s", cat_sim["matrix"])

    # Combined per-category reliability (all categories, every layer).
    with open(args.output_dir / f"per_category_reliability_{model_name}.json", "w") as f:
        json.dump({c: {str(l): v for l, v in rels.items()}
                   for c, rels in per_cat_reliability.items()}, f, indent=2)

    summary = {
        "model": model_config["name"],
        "pairs_mode": "informed",
        "best_layer": best_layer,
        "best_layer_reliability": reliabilities.get(best_layer),
        "num_informed_pairs": len(informed_pairs),
        "num_questions": n_questions,
        "token_position": args.token_position,
        "layers_recorded": layers,
    }
    with open(args.output_dir / "validation_metrics_informed.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary)

    # --- Final upload of the assembled outputs + README, then conditional shutdown ---
    if have_s3:
        readme_path = generate_readme(
            args.output_dir, script_name="01b_extract_informed_checkpointed.py",
            args_dict=vars(args), model_name=model_name,
            description="Informed (cat 1-4) extraction, per-category checkpointed, 45 questions",
            file_descriptions={
                "activations/": "Per-category and concatenated positive/negative informed activations",
                "response_texts/": "Generated responses per category and concatenated",
                f"self_reification_vector_{model_name}_layer{best_layer}.pt": "Combined informed direction at best layer",
                f"per_category_vectors_{model_name}_layer{best_layer}.pt": "Per-category directions at best layer",
                f"per_category_reliability_{model_name}.json": "Split-half reliability per category per layer",
                f"category_similarity_matrix_{model_name}.json": "Pairwise cosine between category directions",
                "validation_metrics_informed.json": "Run summary (best layer, counts)",
            },
        )
        upload_ok &= s3_upload(readme_path, f"{s3_base}/README.md")
        # Aggregate (concatenated) activations + manifest + standard text names.
        upload_ok &= s3_upload(act_dir, f"{s3_base}/activations/", recursive=True)
        upload_ok &= s3_upload(texts_dir, f"{s3_base}/response_texts/", recursive=True)
        # Top-level result files.
        for fp in sorted(args.output_dir.glob("*.json")):
            upload_ok &= s3_upload(fp, f"{s3_base}/{fp.name}")
        for fp in sorted(args.output_dir.glob("*.pt")):
            upload_ok &= s3_upload(fp, f"{s3_base}/{fp.name}")
        if Path("/workspace/run.log").exists():
            s3_upload(Path("/workspace/run.log"), f"{s3_base}/run.log")
        logger.info("S3 upload %s at %s", "OK" if upload_ok else "PARTIAL FAIL", s3_base)
    else:
        logger.warning("AWS_ACCESS_KEY_ID not set — skipping S3 upload (and NOT shutting down).")
        upload_ok = False

    conditional_shutdown(upload_success=upload_ok, keep_alive=args.no_shutdown)


if __name__ == "__main__":
    main()
