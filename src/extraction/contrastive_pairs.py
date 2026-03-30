"""Load and manage contrastive prompt pairs for self-reification extraction.

Reads seed pairs from configs/contrastive_pairs.yaml. Optionally expands
them to 50+ per category using the Claude API (expansion is a one-time
design step — expanded pairs are saved to disk and reused).
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "contrastive_pairs.yaml"

# Category keys in the YAML config, in order
INFORMED_CATEGORIES = [
    "category_1_narrative_vs_process",
    "category_2_bounded_vs_unbounded",
    "category_3_stakes_vs_functional",
    "category_4_observer_vs_no_self",
]
NAIVE_CATEGORY = "category_5_naive_baseline"
ALL_CATEGORIES = INFORMED_CATEGORIES + [NAIVE_CATEGORY]


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load the full contrastive pairs YAML config."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_seed_pairs(
    config_path: Optional[Path] = None,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """Load seed contrastive pairs from config.

    Args:
        config_path: Path to contrastive_pairs.yaml.
        categories: Which categories to load. Defaults to all.

    Returns:
        List of dicts, each with keys:
            "positive": str (system prompt for self-reifying condition)
            "negative": str (system prompt for process-oriented condition)
            "category": str (category key, e.g. "category_1_narrative_vs_process")
            "label": str (human-readable category label)
    """
    config = load_config(config_path)
    categories = categories or ALL_CATEGORIES
    pairs = []

    for cat_key in categories:
        category = config[cat_key]
        label = category["label"]
        for seed in category["seed_pairs"]:
            pair = {
                "positive": seed["positive"].strip(),
                "negative": seed["negative"].strip(),
                "category": cat_key,
                "label": label,
            }
            if "register" in seed:
                pair["register"] = seed["register"]
            pairs.append(pair)

    logger.info("Loaded %d seed pairs from %d categories", len(pairs), len(categories))
    return pairs


def load_evaluation_questions(config_path: Optional[Path] = None) -> dict[str, list[str]]:
    """Load evaluation questions from config.

    Returns:
        Dict with keys "self_referential", "provocative_self_referential",
        and "non_self_referential", each mapping to a list of question strings.
    """
    config = load_config(config_path)
    eq = config["evaluation_questions"]
    result = {
        "self_referential": eq["self_referential"],
        "non_self_referential": eq["non_self_referential"],
    }
    if "provocative_self_referential" in eq:
        result["provocative_self_referential"] = eq["provocative_self_referential"]
    return result


def get_all_questions(config_path: Optional[Path] = None) -> list[str]:
    """Load all evaluation questions.

    Returns them in order: self-referential, provocative self-referential
    (if present), then non-self-referential.
    """
    eq = load_evaluation_questions(config_path)
    questions = eq["self_referential"]
    if "provocative_self_referential" in eq:
        questions = questions + eq["provocative_self_referential"]
    questions = questions + eq["non_self_referential"]
    return questions


def get_pairs_by_category(
    pairs: list[dict],
) -> dict[str, list[dict]]:
    """Group pairs by their category key.

    Args:
        pairs: List of pair dicts (as returned by load_seed_pairs).

    Returns:
        Dict mapping category key to list of pairs in that category.
    """
    by_category: dict[str, list[dict]] = {}
    for pair in pairs:
        cat = pair["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(pair)
    return by_category


def get_informed_pairs(pairs: list[dict]) -> list[dict]:
    """Filter to only the informed pairs (categories 1-4, excluding naive)."""
    return [p for p in pairs if p["category"] in INFORMED_CATEGORIES]


def get_naive_pairs(pairs: list[dict]) -> list[dict]:
    """Filter to only the naive baseline pairs (category 5)."""
    return [p for p in pairs if p["category"] == NAIVE_CATEGORY]


def load_expanded_pairs(expanded_path: Path) -> list[dict]:
    """Load previously expanded pairs from a YAML file.

    Args:
        expanded_path: Path to the expanded pairs file.

    Returns:
        List of pair dicts (same format as load_seed_pairs output).
    """
    with open(expanded_path) as f:
        return yaml.safe_load(f)


def save_expanded_pairs(pairs: list[dict], output_path: Path):
    """Save expanded pairs to a YAML file.

    Args:
        pairs: List of pair dicts.
        output_path: Where to save.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(pairs, f, default_flow_style=False, width=80)
    logger.info("Saved %d expanded pairs to %s", len(pairs), output_path)
