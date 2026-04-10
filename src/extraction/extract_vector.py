"""Extract the self-reification direction using contrastive averaging.

Implements the core persona_vectors methodology:
  1. For each contrastive pair, set the positive (self-reifying) system prompt,
     run all evaluation questions, and record activations. Repeat with the
     negative (process-oriented) system prompt.
  2. Compute mean activation for positive and negative conditions.
  3. Self-reification direction = positive_mean - negative_mean.
  4. Select the best layer by split-half reliability.

Supports extraction per-category (to check construct coherence) and
combined across informed categories (1-4).
"""

import json
import logging
from pathlib import Path

import torch

from extraction.contrastive_pairs import (
    INFORMED_CATEGORIES,
    BASELINE_CATEGORY,
    get_informed_pairs,
    get_baseline_pairs,
    get_pairs_by_category,
    load_seed_pairs,
    get_all_questions,
)
from utils.activation_cache import record_activations, save_activations, load_activations
from utils.metrics import (
    cosine_similarity,
    extract_direction,
    pairwise_cosine_matrix,
    split_half_reliability,
)

logger = logging.getLogger(__name__)


def collect_condition_activations(
    model,
    tokenizer,
    pairs: list[dict],
    questions: list[str],
    layers: list[int],
    max_new_tokens: int = 256,
    token_position: str = "last",
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int]]:
    """Collect activations for positive and negative conditions across all pairs.

    For each pair, runs all evaluation questions under both the positive and
    negative system prompts, recording activations at every specified layer.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        pairs: List of contrastive pair dicts with "positive" and "negative" keys.
        questions: List of evaluation question strings.
        layers: Layer indices to record.
        max_new_tokens: Max response tokens per question.
        token_position: "last" or "mean".

    Returns:
        Tuple of (positive_activations, negative_activations, pair_boundaries,
                  positive_texts, negative_texts).
        Activations: dict mapping layer_idx to tensor of shape
            (num_pairs * num_questions, hidden_size).
        pair_boundaries: list of cumulative sample counts per pair, so we can
            slice per-pair activations later without re-running inference.
            E.g., [30, 60, 90] means pair 0 has samples 0-29, pair 1 has 30-59, etc.
        positive_texts, negative_texts: lists of response strings.
    """
    positive_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    negative_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    pair_boundaries = []
    all_pos_texts: list[str] = []
    all_neg_texts: list[str] = []
    cumulative = 0

    for pair_idx, pair in enumerate(pairs):
        logger.info(
            "Processing pair %d/%d [%s]",
            pair_idx + 1,
            len(pairs),
            pair.get("label", "unknown"),
        )

        # Positive condition
        pos_acts, pos_texts = record_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=questions,
            system_prompt=pair["positive"],
            layers=layers,
            max_new_tokens=max_new_tokens,
            token_position=token_position,
        )
        for l in layers:
            if l in pos_acts:
                positive_per_layer[l].append(pos_acts[l])
        all_pos_texts.extend(pos_texts)

        # Negative condition
        neg_acts, neg_texts = record_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=questions,
            system_prompt=pair["negative"],
            layers=layers,
            max_new_tokens=max_new_tokens,
            token_position=token_position,
        )
        for l in layers:
            if l in neg_acts:
                negative_per_layer[l].append(neg_acts[l])
        all_neg_texts.extend(neg_texts)

        cumulative += len(questions)
        pair_boundaries.append(cumulative)

    # Concatenate across pairs: (total_samples, hidden_size)
    positive = {
        l: torch.cat(tensors, dim=0)
        for l, tensors in positive_per_layer.items()
        if tensors
    }
    negative = {
        l: torch.cat(tensors, dim=0)
        for l, tensors in negative_per_layer.items()
        if tensors
    }

    return positive, negative, pair_boundaries, all_pos_texts, all_neg_texts


def slice_activations_by_pair(
    activations: dict[int, torch.Tensor],
    pair_boundaries: list[int],
    pair_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Slice pre-collected activations to a subset of pairs.

    Args:
        activations: Dict of layer -> (total_samples, hidden_size).
        pair_boundaries: Cumulative sample counts per pair.
        pair_indices: Which pairs to extract (0-indexed).

    Returns:
        Dict of layer -> (subset_samples, hidden_size).
    """
    # Convert boundaries to (start, end) ranges
    ranges = []
    for i in pair_indices:
        start = pair_boundaries[i - 1] if i > 0 else 0
        end = pair_boundaries[i]
        ranges.append((start, end))

    result = {}
    for l, tensor in activations.items():
        slices = [tensor[start:end] for start, end in ranges]
        if slices:
            result[l] = torch.cat(slices, dim=0)
    return result


def extract_all_directions(
    positive: dict[int, torch.Tensor],
    negative: dict[int, torch.Tensor],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Extract direction at each layer from positive and negative activations.

    Args:
        positive: Dict of layer -> (n_pos, hidden_size).
        negative: Dict of layer -> (n_neg, hidden_size).
        layers: Layers to extract from.

    Returns:
        Dict mapping layer index to direction vector (hidden_size,).
    """
    directions = {}
    for l in layers:
        if l in positive and l in negative:
            directions[l] = extract_direction(positive[l], negative[l])
    return directions


def select_best_layer(
    positive: dict[int, torch.Tensor],
    negative: dict[int, torch.Tensor],
    layers: list[int],
    n_splits: int = 100,
) -> tuple[int, dict[int, float]]:
    """Select the layer with highest split-half reliability.

    Args:
        positive: Dict of layer -> activations.
        negative: Dict of layer -> activations.
        layers: Layers to evaluate.
        n_splits: Number of random splits for reliability estimation.

    Returns:
        Tuple of (best_layer_index, dict of layer -> reliability score).
    """
    reliabilities = {}
    for l in layers:
        if l in positive and l in negative:
            rel = split_half_reliability(
                positive[l], negative[l], n_splits=n_splits
            )
            reliabilities[l] = rel
            logger.info("Layer %d split-half reliability: %.4f", l, rel)

    best_layer = max(reliabilities, key=reliabilities.get)
    logger.info(
        "Best layer: %d (reliability=%.4f)", best_layer, reliabilities[best_layer]
    )
    return best_layer, reliabilities


def run_extraction(
    model,
    tokenizer,
    model_config: dict,
    output_dir: Path,
    config_path: Path = None,
    max_new_tokens: int = 256,
    token_position: str = "last",
    n_splits: int = 100,
    resume: bool = False,
    pairs_mode: str = "all",
) -> dict:
    """Run the full extraction pipeline (Experiment 1.1).

    Steps:
      1. Load contrastive pairs and evaluation questions.
      2. Collect activations for positive/negative conditions at all layers.
      3. Extract combined (cat 1-4) and per-category directions.
      4. Extract baseline (cat 5) direction.
      5. Select best layer by split-half reliability.
      6. Compute within-category similarity and baseline-vs-informed comparison.
      7. Save all results.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        model_config: Config dict from model_loader (has num_layers, name, etc.).
        output_dir: Where to save results (e.g., data/results/1.1/).
        config_path: Path to contrastive_pairs.yaml.
        max_new_tokens: Max response length.
        token_position: "last" or "mean".
        n_splits: Number of splits for reliability estimation.
        pairs_mode: "all" (default), "informed" (categories 1-4 only),
            or "baseline" (category 5 only). "naive" accepted as alias. Use for parallel runs on separate GPUs.

    Returns:
        Summary dict with key results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_layers = model_config["num_layers"]
    # For large models, record every Nth layer to reduce memory.
    # Default stride=1 (all layers). For 72B+ models, stride=4 recommended.
    layer_stride = model_config.get("layer_stride", 1)
    layers = list(range(0, num_layers, layer_stride))
    if layer_stride > 1:
        # Always include the last layer
        if (num_layers - 1) not in layers:
            layers.append(num_layers - 1)
        logger.info("Using layer stride %d: recording %d of %d layers",
                     layer_stride, len(layers), num_layers)
    model_name = model_config["name"].replace("/", "_")

    run_informed = pairs_mode in ("all", "informed")
    run_baseline = pairs_mode in ("all", "baseline", "naive")

    # Step 1: Load pairs and questions
    all_pairs = load_seed_pairs(config_path)
    informed_pairs = get_informed_pairs(all_pairs)
    baseline_pairs = get_baseline_pairs(all_pairs)
    questions = get_all_questions(config_path)

    logger.info(
        "Loaded %d informed pairs, %d baseline pairs, %d questions (mode=%s)",
        len(informed_pairs),
        len(baseline_pairs),
        len(questions),
        pairs_mode,
    )

    best_layer = None
    reliabilities = {}
    combined_dir = None
    per_cat_reliability = {}

    # Step 2-4: Informed pairs (categories 1-4)
    if run_informed:
        activations_dir = output_dir / "activations"
        pos_checkpoint = load_activations(activations_dir, f"positive_informed_{model_name}", layers)
        neg_checkpoint = load_activations(activations_dir, f"negative_informed_{model_name}", layers)

        if resume and len(pos_checkpoint) == len(layers) and len(neg_checkpoint) == len(layers):
            logger.info("=== Resuming: loaded cached informed activations (%d layers) ===", len(layers))
            pos_informed = pos_checkpoint
            neg_informed = neg_checkpoint
            n_questions = len(questions)
            pair_boundaries = [n_questions * (i + 1) for i in range(len(informed_pairs))]
        else:
            logger.info("=== Collecting activations for informed pairs ===")
            pos_informed, neg_informed, pair_boundaries, _, _ = collect_condition_activations(
                model, tokenizer, informed_pairs, questions, layers,
                max_new_tokens=max_new_tokens, token_position=token_position,
            )
            save_activations(pos_informed, activations_dir, f"positive_informed_{model_name}")
            save_activations(neg_informed, activations_dir, f"negative_informed_{model_name}")

        # Extract combined direction and select best layer
        logger.info("=== Extracting combined direction ===")
        combined_directions = extract_all_directions(pos_informed, neg_informed, layers)
        best_layer, reliabilities = select_best_layer(
            pos_informed, neg_informed, layers, n_splits=n_splits
        )

        combined_dir = combined_directions[best_layer]
        torch.save(
            combined_dir,
            output_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt",
        )

        with open(output_dir / f"layer_reliability_{model_name}.json", "w") as f:
            json.dump(
                {"reliabilities": {str(k): v for k, v in reliabilities.items()},
                 "best_layer": best_layer},
                f, indent=2,
            )

        # Per-category extraction at the best layer
        logger.info("=== Extracting per-category directions (from cached activations) ===")
        pairs_by_cat = get_pairs_by_category(informed_pairs)
        per_category_vectors = {}

        pair_idx = 0
        cat_pair_indices: dict[str, list[int]] = {k: [] for k in INFORMED_CATEGORIES}
        for p in informed_pairs:
            cat_pair_indices[p["category"]].append(pair_idx)
            pair_idx += 1

        for cat_key in INFORMED_CATEGORIES:
            indices = cat_pair_indices[cat_key]
            if not indices:
                continue
            pos_cat = slice_activations_by_pair(pos_informed, pair_boundaries, indices)
            neg_cat = slice_activations_by_pair(neg_informed, pair_boundaries, indices)
            if best_layer in pos_cat and best_layer in neg_cat:
                cat_dir = extract_direction(pos_cat[best_layer], neg_cat[best_layer])
                per_category_vectors[cat_key] = cat_dir

        torch.save(
            per_category_vectors,
            output_dir / f"per_category_vectors_{model_name}_layer{best_layer}.pt",
        )

        if len(per_category_vectors) > 1:
            cat_sim = pairwise_cosine_matrix(per_category_vectors)
            with open(output_dir / f"category_similarity_matrix_{model_name}.json", "w") as f:
                json.dump(cat_sim, f, indent=2)
            logger.info("Category similarity matrix: %s", cat_sim["matrix"])

        # Per-category split-half reliability at every layer
        logger.info("=== Computing per-category reliability by layer ===")
        for cat_key in INFORMED_CATEGORIES:
            indices = cat_pair_indices[cat_key]
            if not indices:
                continue
            pos_cat = slice_activations_by_pair(pos_informed, pair_boundaries, indices)
            neg_cat = slice_activations_by_pair(neg_informed, pair_boundaries, indices)
            cat_rel = {}
            for l in layers:
                if l in pos_cat and l in neg_cat:
                    cat_rel[l] = split_half_reliability(pos_cat[l], neg_cat[l], n_splits=n_splits)
            per_cat_reliability[cat_key] = cat_rel
            logger.info("  %s: best layer %d (%.4f)",
                         cat_key, max(cat_rel, key=cat_rel.get), max(cat_rel.values()))

        with open(output_dir / f"per_category_reliability_{model_name}.json", "w") as f:
            json.dump(
                {cat: {str(l): v for l, v in rels.items()}
                 for cat, rels in per_cat_reliability.items()},
                f, indent=2,
            )

    # Step 5: Naive baseline extraction
    baseline_best_layer = None
    baseline_reliability = {}
    baseline_vs_informed_cosine = None

    if run_baseline:
        logger.info("=== Extracting baseline direction ===")
        activations_dir = output_dir / "activations"
        pos_baseline, neg_baseline, _, pos_baseline_texts, neg_baseline_texts = collect_condition_activations(
            model, tokenizer, baseline_pairs, questions, layers,
            max_new_tokens=max_new_tokens, token_position=token_position,
        )

        # Save baseline activations and response texts for later analysis
        save_activations(pos_baseline, activations_dir, f"positive_baseline_{model_name}")
        save_activations(neg_baseline, activations_dir, f"negative_baseline_{model_name}")

        # Save response texts for pronoun density and qualitative analysis
        import json as _json
        texts_dir = output_dir / "response_texts"
        texts_dir.mkdir(parents=True, exist_ok=True)
        with open(texts_dir / f"positive_baseline_{model_name}.json", "w") as f:
            _json.dump(pos_baseline_texts, f)
        with open(texts_dir / f"negative_baseline_{model_name}.json", "w") as f:
            _json.dump(neg_baseline_texts, f)
        logger.info("Saved %d response texts per condition", len(pos_baseline_texts))

        # Naive split-half reliability by layer
        for l in layers:
            if l in pos_baseline and l in neg_baseline:
                baseline_reliability[l] = split_half_reliability(
                    pos_baseline[l], neg_baseline[l], n_splits=n_splits
                )

        baseline_best_layer = max(baseline_reliability, key=baseline_reliability.get) if baseline_reliability else 0
        logger.info("Naive best layer: %d (reliability=%.4f)",
                     baseline_best_layer, baseline_reliability.get(baseline_best_layer, 0))

        # Save baseline direction at baseline best layer (and informed best layer if available)
        save_layers = {baseline_best_layer}
        if best_layer is not None:
            save_layers.add(best_layer)

        for save_layer in sorted(save_layers):
            if save_layer in pos_baseline and save_layer in neg_baseline:
                direction = extract_direction(pos_baseline[save_layer], neg_baseline[save_layer])
                torch.save(
                    direction,
                    output_dir / f"baseline_vector_{model_name}_layer{save_layer}.pt",
                )

        # Compare baseline vs informed if both were run
        if combined_dir is not None and best_layer in pos_baseline and best_layer in neg_baseline:
            baseline_at_informed_layer = extract_direction(pos_baseline[best_layer], neg_baseline[best_layer])
            baseline_vs_informed_cosine = cosine_similarity(combined_dir, baseline_at_informed_layer)
            with open(output_dir / f"baseline_vs_informed_cosine_{model_name}.json", "w") as f:
                json.dump({"cosine_similarity": baseline_vs_informed_cosine}, f, indent=2)
            logger.info("Naive vs informed cosine similarity: %.4f", baseline_vs_informed_cosine)

        # Per-register analysis (if register tags are present)
        register_groups: dict[str, list[int]] = {}
        for i, pair in enumerate(baseline_pairs):
            reg = pair.get("register", "untagged")
            register_groups.setdefault(reg, []).append(i)

        if len(register_groups) > 1:
            logger.info("=== Per-register analysis (%d registers) ===", len(register_groups))

            # Reconstruct pair boundaries for baseline pairs
            n_questions = len(questions)
            baseline_boundaries = [n_questions * (i + 1) for i in range(len(baseline_pairs))]

            per_register_vectors = {}
            per_register_reliability = {}

            for reg_name, indices in register_groups.items():
                pos_reg = slice_activations_by_pair(pos_baseline, baseline_boundaries, indices)
                neg_reg = slice_activations_by_pair(neg_baseline, baseline_boundaries, indices)

                # Reliability at baseline best layer
                if baseline_best_layer in pos_reg and baseline_best_layer in neg_reg:
                    reg_rel = split_half_reliability(
                        pos_reg[baseline_best_layer], neg_reg[baseline_best_layer], n_splits=n_splits
                    )
                    per_register_reliability[reg_name] = reg_rel
                    logger.info("  %s: reliability=%.4f (layer %d)", reg_name, reg_rel, baseline_best_layer)

                    # Extract direction at baseline best layer
                    reg_dir = extract_direction(pos_reg[baseline_best_layer], neg_reg[baseline_best_layer])
                    per_register_vectors[reg_name] = reg_dir

            # Cross-register cosine similarity
            if len(per_register_vectors) > 1:
                reg_sim = pairwise_cosine_matrix(per_register_vectors)
                with open(output_dir / f"register_similarity_matrix_{model_name}.json", "w") as f:
                    json.dump(reg_sim, f, indent=2)
                logger.info("Register similarity matrix: %s", reg_sim["matrix"])

            # Save per-register vectors and reliability
            if per_register_vectors:
                torch.save(
                    per_register_vectors,
                    output_dir / f"per_register_vectors_{model_name}_layer{baseline_best_layer}.pt",
                )
            if per_register_reliability:
                with open(output_dir / f"per_register_reliability_{model_name}.json", "w") as f:
                    json.dump(per_register_reliability, f, indent=2)

        # Save baseline reliability (append to per-category if informed was also run)
        if baseline_reliability:
            per_cat_reliability["baseline"] = baseline_reliability
            with open(output_dir / f"per_category_reliability_{model_name}.json", "w") as f:
                json.dump(
                    {cat: {str(l): v for l, v in rels.items()}
                     for cat, rels in per_cat_reliability.items()},
                    f, indent=2,
                )

    # Step 6: Summary
    summary = {
        "model": model_config["name"],
        "pairs_mode": pairs_mode,
        "best_layer": best_layer,
        "best_layer_reliability": reliabilities.get(best_layer, None) if best_layer else None,
        "baseline_best_layer": baseline_best_layer,
        "baseline_best_layer_reliability": baseline_reliability.get(baseline_best_layer, None) if baseline_best_layer else None,
        "num_informed_pairs": len(informed_pairs) if run_informed else 0,
        "num_baseline_pairs": len(baseline_pairs) if run_baseline else 0,
        "num_questions": len(questions),
        "baseline_vs_informed_cosine": baseline_vs_informed_cosine,
        "token_position": token_position,
    }

    with open(output_dir / f"validation_metrics_{pairs_mode}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Extraction complete ===")
    logger.info("Summary: %s", summary)
    return summary
