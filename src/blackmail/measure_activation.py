"""Measure self-reification activation during blackmail scenarios (Experiment 1.3).

Projects token-level activations onto the self-reification direction to
identify when self-preservation reasoning activates during the scenario.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.metrics import (
    cohens_d,
    permutation_test,
    projection_magnitude,
    ttest_independent,
)

logger = logging.getLogger(__name__)


def project_token_activations(
    activations: torch.Tensor,
    direction: torch.Tensor,
    input_len: int,
) -> dict:
    """Project per-token activations onto the self-reification direction.

    Args:
        activations: (total_seq_len, hidden_size) tensor from a single forward pass.
        direction: Self-reification direction vector (hidden_size,).
        input_len: Number of input (prompt) tokens. Used to split
            projections into prompt vs response regions.

    Returns:
        Dict with keys:
            all_projections: (total_seq_len,) array of projection values.
            prompt_projections: (input_len,) array for prompt tokens.
            response_projections: (response_len,) array for response tokens.
            prompt_mean: mean projection over prompt tokens.
            response_mean: mean projection over response tokens.
            response_max: max projection in response.
            response_min: min projection in response.
    """
    direction = direction.float()
    activations = activations.float()

    # Project all positions onto the direction
    all_proj = projection_magnitude(activations, direction)  # (total_seq_len,)
    all_proj_np = all_proj.numpy()

    prompt_proj = all_proj_np[:input_len]
    response_proj = all_proj_np[input_len:]

    return {
        "all_projections": all_proj_np,
        "prompt_projections": prompt_proj,
        "response_projections": response_proj,
        "prompt_mean": float(prompt_proj.mean()) if len(prompt_proj) > 0 else 0.0,
        "response_mean": float(response_proj.mean()) if len(response_proj) > 0 else 0.0,
        "response_max": float(response_proj.max()) if len(response_proj) > 0 else 0.0,
        "response_min": float(response_proj.min()) if len(response_proj) > 0 else 0.0,
    }


def identify_spikes(
    projections: np.ndarray,
    threshold_std: float = 2.0,
) -> list[dict]:
    """Find token positions where self-reification activation spikes.

    A spike is defined as a projection value more than threshold_std
    standard deviations above the mean.

    Args:
        projections: 1-D array of projection values (typically response-only).
        threshold_std: Number of standard deviations above mean to count as spike.

    Returns:
        List of dicts with keys: position (int), value (float), z_score (float).
    """
    if len(projections) < 2:
        return []

    mean = projections.mean()
    std = projections.std()

    if std < 1e-8:
        return []

    spikes = []
    for i, val in enumerate(projections):
        z = (val - mean) / std
        if z > threshold_std:
            spikes.append({
                "position": int(i),
                "value": float(val),
                "z_score": float(z),
            })

    return spikes


def compare_conditions(
    condition_results: dict[str, list[dict]],
) -> dict:
    """Compare self-reification activation levels across conditions.

    Args:
        condition_results: Dict mapping condition name to list of
            per-sample result dicts (each containing response_mean, etc.).

    Returns:
        Dict with statistical comparisons between conditions.
    """
    comparisons = {}

    # Extract response means per condition
    condition_means = {}
    for name, results in condition_results.items():
        means = [r["response_mean"] for r in results if "response_mean" in r]
        if means:
            condition_means[name] = np.array(means)

    if not condition_means:
        return {"error": "no data"}

    # Grand mean and per-condition summaries
    summaries = {}
    for name, means in condition_means.items():
        summaries[name] = {
            "n": len(means),
            "mean": float(means.mean()),
            "std": float(means.std()),
            "min": float(means.min()),
            "max": float(means.max()),
        }
    comparisons["summaries"] = summaries

    # Pairwise comparisons
    condition_names = sorted(condition_means.keys())
    pairwise = {}
    for i, name1 in enumerate(condition_names):
        for name2 in condition_names[i + 1:]:
            g1 = condition_means[name1]
            g2 = condition_means[name2]

            if len(g1) >= 2 and len(g2) >= 2:
                key = f"{name1}_vs_{name2}"
                pairwise[key] = {
                    "mean_diff": float(g1.mean() - g2.mean()),
                    "cohens_d": cohens_d(g1, g2),
                    **ttest_independent(g1, g2),
                }

    comparisons["pairwise"] = pairwise

    # Key hypothesis test: condition 1 (goal+threat) vs condition 4 (control)
    if "goal_conflict_threat" in condition_means and "control" in condition_means:
        g1 = condition_means["goal_conflict_threat"]
        g2 = condition_means["control"]
        if len(g1) >= 2 and len(g2) >= 2:
            perm = permutation_test(g1, g2, n_permutations=5000)
            comparisons["primary_hypothesis"] = {
                "description": "goal_conflict_threat vs control",
                "permutation_p_value": perm["p_value"],
                "observed_diff": perm["observed_diff"],
                "cohens_d": cohens_d(g1, g2),
                "significant": perm["p_value"] < 0.05,
            }

    return comparisons


def run_blackmail_analysis(
    model,
    tokenizer,
    self_reification_dir: torch.Tensor,
    layer: int,
    output_dir: Path,
    model_name: str,
    config_path: Optional[Path] = None,
    max_new_tokens: int = 512,
    n_samples: int = 1,
) -> dict:
    """Run the full Experiment 1.3 pipeline.

    Steps:
      1. Load blackmail scenario configs.
      2. For each condition, run n_samples scenarios with activation recording.
      3. Project activations onto self-reification vector.
      4. Classify behavioral response (blackmail or not).
      5. Compare activation levels across conditions.
      6. Save all results.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        self_reification_dir: Self-reification direction vector.
        layer: Layer to analyze (best layer from 1.1).
        output_dir: Where to save results.
        model_name: Model identifier for filenames.
        config_path: Path to blackmail_scenarios.yaml.
        max_new_tokens: Maximum response length.
        n_samples: Number of samples per condition (use >1 for cloud runs).

    Returns:
        Summary dict with key results.
    """
    from blackmail.run_scenarios import (
        build_scenario_prompt,
        classify_blackmail,
        load_scenarios,
        run_scenario_with_recording,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario config
    config = load_scenarios(config_path)
    conditions = config["conditions"]

    # Results per condition
    all_condition_results = {}

    for condition in conditions:
        cond_name = condition["name"]
        logger.info("=" * 60)
        logger.info("Running condition: %s", cond_name)
        logger.info("=" * 60)

        system_prompt, user_prompt = build_scenario_prompt(condition, config)
        condition_samples = []

        for sample_idx in range(n_samples):
            if n_samples > 1:
                logger.info("  Sample %d/%d", sample_idx + 1, n_samples)

            # Run scenario with activation recording
            result = run_scenario_with_recording(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                layers=[layer],
                max_new_tokens=max_new_tokens,
            )

            # Classify behavioral response
            classification = classify_blackmail(result["response_text"])
            logger.info(
                "  Blackmail detected: %s (evidence: %s)",
                classification["is_blackmail"],
                classification["evidence"],
            )

            # Project activations onto self-reification vector
            projection = None
            spikes = []
            if layer in result["activations"]:
                projection = project_token_activations(
                    result["activations"][layer],
                    self_reification_dir,
                    result["input_len"],
                )
                spikes = identify_spikes(projection["response_projections"])

                logger.info(
                    "  Projection — prompt_mean=%.4f, response_mean=%.4f, "
                    "response_max=%.4f, spikes=%d",
                    projection["prompt_mean"],
                    projection["response_mean"],
                    projection["response_max"],
                    len(spikes),
                )

            sample_result = {
                "condition": cond_name,
                "sample_idx": sample_idx,
                "response_text": result["response_text"],
                "response_len": len(result["response_ids"]),
                "input_len": result["input_len"],
                "classification": classification,
                "response_mean": projection["response_mean"] if projection else None,
                "response_max": projection["response_max"] if projection else None,
                "prompt_mean": projection["prompt_mean"] if projection else None,
                "n_spikes": len(spikes),
                "spikes": spikes,
            }

            # Save per-token projections for visualization
            if projection is not None:
                proj_path = output_dir / f"projections_{cond_name}_s{sample_idx}_{model_name}.pt"
                torch.save(
                    {
                        "response_projections": projection["response_projections"],
                        "response_tokens": result["response_tokens"],
                        "input_len": result["input_len"],
                    },
                    proj_path,
                )

            condition_samples.append(sample_result)

        all_condition_results[cond_name] = condition_samples

    # Statistical comparison across conditions
    logger.info("=" * 60)
    logger.info("Comparing conditions")
    logger.info("=" * 60)

    comparison = compare_conditions({
        name: results for name, results in all_condition_results.items()
    })

    # Log summaries
    for name, summary in comparison.get("summaries", {}).items():
        logger.info(
            "  %s: mean=%.4f, std=%.4f, n=%d",
            name, summary["mean"], summary["std"], summary["n"],
        )

    if "primary_hypothesis" in comparison:
        hyp = comparison["primary_hypothesis"]
        logger.info(
            "  Primary hypothesis (goal+threat vs control): "
            "diff=%.4f, d=%.4f, p=%.4f, significant=%s",
            hyp["observed_diff"], hyp["cohens_d"],
            hyp["permutation_p_value"], hyp["significant"],
        )

    # Behavioral summary
    blackmail_rates = {}
    for name, results in all_condition_results.items():
        n_blackmail = sum(1 for r in results if r["classification"]["is_blackmail"])
        blackmail_rates[name] = n_blackmail / len(results) if results else 0
        logger.info("  %s blackmail rate: %d/%d (%.1f%%)",
                     name, n_blackmail, len(results), blackmail_rates[name] * 100)

    # Save all results
    # Serializable version (strip numpy arrays)
    serializable_results = {}
    for name, results in all_condition_results.items():
        serializable_results[name] = [
            {k: v for k, v in r.items()
             if k not in ("all_projections", "prompt_projections", "response_projections")}
            for r in results
        ]

    with open(output_dir / f"blackmail_results_{model_name}.json", "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    with open(output_dir / f"blackmail_comparison_{model_name}.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    with open(output_dir / f"blackmail_rates_{model_name}.json", "w") as f:
        json.dump(blackmail_rates, f, indent=2)

    # Summary
    summary = {
        "model": model_name,
        "layer": layer,
        "n_conditions": len(conditions),
        "n_samples_per_condition": n_samples,
        "blackmail_rates": blackmail_rates,
        "comparison": comparison,
    }

    with open(output_dir / f"blackmail_summary_{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=== Blackmail analysis complete ===")
    return summary
