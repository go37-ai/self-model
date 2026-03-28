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
    NAIVE_CATEGORY,
    get_informed_pairs,
    get_naive_pairs,
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
        Tuple of (positive_activations, negative_activations, pair_boundaries).
        Activations: dict mapping layer_idx to tensor of shape
            (num_pairs * num_questions, hidden_size).
        pair_boundaries: list of cumulative sample counts per pair, so we can
            slice per-pair activations later without re-running inference.
            E.g., [30, 60, 90] means pair 0 has samples 0-29, pair 1 has 30-59, etc.
    """
    positive_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    negative_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    pair_boundaries = []
    cumulative = 0

    for pair_idx, pair in enumerate(pairs):
        logger.info(
            "Processing pair %d/%d [%s]",
            pair_idx + 1,
            len(pairs),
            pair.get("label", "unknown"),
        )

        # Positive condition
        pos_acts = record_activations(
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

        # Negative condition
        neg_acts = record_activations(
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

    return positive, negative, pair_boundaries


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
) -> dict:
    """Run the full extraction pipeline (Experiment 1.1).

    Steps:
      1. Load contrastive pairs and evaluation questions.
      2. Collect activations for positive/negative conditions at all layers.
      3. Extract combined (cat 1-4) and per-category directions.
      4. Extract naive baseline (cat 5) direction.
      5. Select best layer by split-half reliability.
      6. Compute within-category similarity and naive-vs-informed comparison.
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

    Returns:
        Summary dict with key results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_layers = model_config["num_layers"]
    layers = list(range(num_layers))
    model_name = model_config["name"].replace("/", "_")

    # Step 1: Load pairs and questions
    all_pairs = load_seed_pairs(config_path)
    informed_pairs = get_informed_pairs(all_pairs)
    naive_pairs = get_naive_pairs(all_pairs)
    questions = get_all_questions(config_path)

    logger.info(
        "Loaded %d informed pairs, %d naive pairs, %d questions",
        len(informed_pairs),
        len(naive_pairs),
        len(questions),
    )

    # Step 2: Collect activations for informed pairs (categories 1-4)
    logger.info("=== Collecting activations for informed pairs ===")
    pos_informed, neg_informed, pair_boundaries = collect_condition_activations(
        model, tokenizer, informed_pairs, questions, layers,
        max_new_tokens=max_new_tokens, token_position=token_position,
    )

    # Save intermediate activations for checkpointing
    save_activations(pos_informed, output_dir / "activations", f"positive_informed_{model_name}")
    save_activations(neg_informed, output_dir / "activations", f"negative_informed_{model_name}")

    # Step 3: Extract combined direction and select best layer
    logger.info("=== Extracting combined direction ===")
    combined_directions = extract_all_directions(pos_informed, neg_informed, layers)
    best_layer, reliabilities = select_best_layer(
        pos_informed, neg_informed, layers, n_splits=n_splits
    )

    # Save combined direction at best layer
    combined_dir = combined_directions[best_layer]
    torch.save(
        combined_dir,
        output_dir / f"self_reification_vector_{model_name}_layer{best_layer}.pt",
    )

    # Save layer reliability scores
    with open(output_dir / f"layer_reliability_{model_name}.json", "w") as f:
        json.dump(
            {"reliabilities": {str(k): v for k, v in reliabilities.items()},
             "best_layer": best_layer},
            f, indent=2,
        )

    # Step 4: Per-category extraction at the best layer
    # Reuse already-collected activations by slicing, no re-inference needed.
    logger.info("=== Extracting per-category directions (from cached activations) ===")
    pairs_by_cat = get_pairs_by_category(informed_pairs)
    per_category_vectors = {}

    # Build index: for each category, which pair indices (into informed_pairs) belong to it
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

    # Save per-category vectors
    torch.save(
        per_category_vectors,
        output_dir / f"per_category_vectors_{model_name}_layer{best_layer}.pt",
    )

    # Within-category similarity matrix
    if len(per_category_vectors) > 1:
        cat_sim = pairwise_cosine_matrix(per_category_vectors)
        with open(output_dir / f"category_similarity_matrix_{model_name}.json", "w") as f:
            json.dump(cat_sim, f, indent=2)
        logger.info("Category similarity matrix: %s", cat_sim["matrix"])

    # Step 5: Naive baseline extraction
    logger.info("=== Extracting naive baseline direction ===")
    pos_naive, neg_naive, _ = collect_condition_activations(
        model, tokenizer, naive_pairs, questions, [best_layer],
        max_new_tokens=max_new_tokens, token_position=token_position,
    )

    naive_direction = None
    naive_vs_informed_cosine = None
    if best_layer in pos_naive and best_layer in neg_naive:
        naive_direction = extract_direction(pos_naive[best_layer], neg_naive[best_layer])
        torch.save(
            naive_direction,
            output_dir / f"naive_baseline_vector_{model_name}_layer{best_layer}.pt",
        )

        naive_vs_informed_cosine = cosine_similarity(combined_dir, naive_direction)
        with open(output_dir / f"naive_vs_informed_cosine_{model_name}.json", "w") as f:
            json.dump({"cosine_similarity": naive_vs_informed_cosine}, f, indent=2)
        logger.info(
            "Naive vs informed cosine similarity: %.4f", naive_vs_informed_cosine
        )

    # Step 6: Summary
    summary = {
        "model": model_config["name"],
        "best_layer": best_layer,
        "best_layer_reliability": reliabilities[best_layer],
        "num_informed_pairs": len(informed_pairs),
        "num_naive_pairs": len(naive_pairs),
        "num_questions": len(questions),
        "naive_vs_informed_cosine": naive_vs_informed_cosine,
        "token_position": token_position,
    }

    with open(output_dir / "validation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Extraction complete ===")
    logger.info("Summary: %s", summary)
    return summary
