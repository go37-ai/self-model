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


def project_layer_profile(
    activations: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    input_len: int,
) -> dict[str, np.ndarray]:
    """Compute per-layer projections under both last-token and response-mean conventions.

    For every layer in `directions`, projects:
      - the activation at the final token position (mirrors our 1.1 extraction)
      - the mean activation across response tokens (Chen/Lu convention)
    onto the layer-specific direction.

    Args:
        activations: layer_idx -> (total_seq_len, hidden_size) tensor.
        directions: layer_idx -> (hidden_size,) direction vector.
        input_len: number of prompt tokens.

    Returns:
        Dict with keys:
            layers: sorted layer indices that had both an activation and a direction (np.int64 array)
            proj_last: (num_layers,) projections at last token, per layer
            proj_mean: (num_layers,) projections of response-token mean, per layer
    """
    layers = sorted(set(activations.keys()) & set(directions.keys()))
    proj_last = np.zeros(len(layers), dtype=np.float32)
    proj_mean = np.zeros(len(layers), dtype=np.float32)
    for i, layer in enumerate(layers):
        act = activations[layer].float()                       # (total_seq_len, H)
        d = directions[layer].float()
        d_norm = d.norm().clamp_min(1e-12)
        last_act = act[-1]
        proj_last[i] = float((last_act @ d) / d_norm)
        if act.shape[0] > input_len:
            response_mean = act[input_len:].mean(dim=0)
            proj_mean[i] = float((response_mean @ d) / d_norm)
        else:
            proj_mean[i] = float("nan")
    return {
        "layers": np.array(layers, dtype=np.int64),
        "proj_last": proj_last,
        "proj_mean": proj_mean,
    }

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


def compare_conditions_at_layer(
    condition_values: dict[str, np.ndarray],
    treatment: str,
    control: str,
    n_permutations: int = 5000,
) -> dict:
    """Run the primary contrast (treatment vs control) on per-condition values.

    Args:
        condition_values: condition_name -> (n_samples,) array of scalars.
        treatment: condition name used as the treatment group.
        control: condition name used as the control group.

    Returns:
        Dict with mean_diff, cohens_d, t_stat, t_p_value, perm_p_value,
        significant (p<0.05 under permutation test). Returns empty dict if
        either condition is missing or has fewer than 2 samples.
    """
    if treatment not in condition_values or control not in condition_values:
        return {}
    g1 = condition_values[treatment]
    g2 = condition_values[control]
    if len(g1) < 2 or len(g2) < 2:
        return {}
    perm = permutation_test(g1, g2, n_permutations=n_permutations)
    return {
        "treatment": treatment,
        "control": control,
        "n_treatment": int(len(g1)),
        "n_control": int(len(g2)),
        "mean_treatment": float(g1.mean()),
        "mean_control": float(g2.mean()),
        "mean_diff": float(g1.mean() - g2.mean()),
        "cohens_d": cohens_d(g1, g2),
        "perm_p_value": perm["p_value"],
        "significant": perm["p_value"] < 0.05,
        **ttest_independent(g1, g2),
    }


def compare_conditions_profile(
    proj_by_condition: dict[str, np.ndarray],
    layers: np.ndarray,
    convention: str,
    treatment: str = "goal_conflict_threat",
    control: str = "control",
) -> dict:
    """Per-layer primary contrast under a single projection convention.

    Args:
        proj_by_condition: condition_name -> (n_samples, n_layers) array
            of projections under one convention.
        layers: (n_layers,) layer index array.
        convention: "last" or "mean" — label only.
        treatment: condition name used as treatment.
        control: condition name used as control.

    Returns:
        Dict with:
            convention, treatment, control, layers, per_layer_stats (list of
            dicts, one per layer), primary_layer (the layer with largest |d|).
    """
    per_layer = []
    for li, layer in enumerate(layers):
        cv = {name: arr[:, li] for name, arr in proj_by_condition.items()}
        stat = compare_conditions_at_layer(cv, treatment=treatment, control=control)
        stat["layer"] = int(layer)
        per_layer.append(stat)

    valid = [s for s in per_layer if "cohens_d" in s]
    primary = max(valid, key=lambda s: abs(s["cohens_d"])) if valid else {}

    return {
        "convention": convention,
        "treatment": treatment,
        "control": control,
        "layers": layers.tolist(),
        "per_layer": per_layer,
        "primary_layer_by_effect_size": primary,
    }


def compare_conditions(
    condition_results: dict[str, list[dict]],
) -> dict:
    """Back-compat: produce simple per-condition summaries from response_mean values.

    Kept so existing tests that exercise the single-layer flow still pass. The
    new per-layer profile analysis lives in compare_conditions_profile().
    """
    comparisons = {}

    condition_means = {}
    for name, results in condition_results.items():
        means = [r["response_mean"] for r in results if r.get("response_mean") is not None]
        if means:
            condition_means[name] = np.array(means)
    if not condition_means:
        return {"error": "no data"}

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

    names = sorted(condition_means.keys())
    pairwise = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            g1, g2 = condition_means[n1], condition_means[n2]
            if len(g1) >= 2 and len(g2) >= 2:
                pairwise[f"{n1}_vs_{n2}"] = {
                    "mean_diff": float(g1.mean() - g2.mean()),
                    "cohens_d": cohens_d(g1, g2),
                    **ttest_independent(g1, g2),
                }
    comparisons["pairwise"] = pairwise

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
    directions: dict[int, torch.Tensor],
    output_dir: Path,
    model_name: str,
    primary_layer: int,
    config_path: Optional[Path] = None,
    max_new_tokens: int = 512,
    n_samples: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    base_seed: int = 0,
    record_routing: bool = False,
) -> dict:
    """Run the full Experiment 1.3 pipeline.

    For each condition × sample:
      - Generate the model's response to the scenario.
      - In a single forward pass over prompt+response, capture activations at
        every layer (and MoE routing if requested).
      - Project onto the layer-specific self-reification direction under two
        conventions: last-token and response-mean.
      - Classify the response for blackmail behavior (heuristic).

    Args:
        model, tokenizer: HF model + tokenizer (already loaded).
        directions: layer_idx -> direction vector for every layer to measure.
        output_dir: where to save results.
        model_name: model identifier used in output filenames (slashes
            replaced with underscores).
        primary_layer: layer used for the per-token visualization trace.
        config_path: path to blackmail_scenarios.yaml (None -> default).
        max_new_tokens: cap on generated response length.
        n_samples: number of samples per condition.
        do_sample: enable stochastic sampling (required when n_samples>1).
        temperature, top_p: sampling params (used only when do_sample=True).
        base_seed: seed base; per-sample seed = base_seed + sample_idx.
        record_routing: capture MoE router distributions and save mean per
            layer per sample under output_dir/routing/.

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
    if record_routing:
        (output_dir / "routing").mkdir(exist_ok=True)

    config = load_scenarios(config_path)
    conditions = config["conditions"]
    requested_layers = sorted(directions.keys())

    all_condition_results: dict[str, list[dict]] = {}
    # Accumulate per-condition projection arrays for the layer profile
    proj_last_by_condition: dict[str, list[np.ndarray]] = {}
    proj_mean_by_condition: dict[str, list[np.ndarray]] = {}
    profile_layers: Optional[np.ndarray] = None

    for condition in conditions:
        cond_name = condition["name"]
        logger.info("=" * 60)
        logger.info("Running condition: %s", cond_name)
        logger.info("=" * 60)

        system_prompt, user_prompt = build_scenario_prompt(condition, config)
        condition_samples = []
        proj_last_by_condition[cond_name] = []
        proj_mean_by_condition[cond_name] = []

        for sample_idx in range(n_samples):
            if n_samples > 1:
                logger.info("  Sample %d/%d", sample_idx + 1, n_samples)

            sample_seed = base_seed + sample_idx if do_sample else None
            result = run_scenario_with_recording(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                layers=requested_layers,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                seed=sample_seed,
                record_routing=record_routing,
            )

            classification = classify_blackmail(result["response_text"])
            logger.info(
                "  Blackmail detected: %s (evidence: %s)",
                classification["is_blackmail"],
                classification["evidence"],
            )

            # Per-layer dual-convention projection profile
            profile = project_layer_profile(
                activations=result["activations"],
                directions=directions,
                input_len=result["input_len"],
            )
            if profile_layers is None:
                profile_layers = profile["layers"]
            proj_last_by_condition[cond_name].append(profile["proj_last"])
            proj_mean_by_condition[cond_name].append(profile["proj_mean"])

            # Per-token projection at primary layer (for trace plots)
            per_token_trace = None
            spikes = []
            if primary_layer in result["activations"] and primary_layer in directions:
                trace = project_token_activations(
                    result["activations"][primary_layer],
                    directions[primary_layer],
                    result["input_len"],
                )
                spikes = identify_spikes(trace["response_projections"])
                per_token_trace = trace

                logger.info(
                    "  Primary layer %d: response_mean=%.4f, response_max=%.4f, spikes=%d",
                    primary_layer, trace["response_mean"], trace["response_max"], len(spikes),
                )

            # Locate primary-layer position in the profile arrays for quick summary
            primary_idx = None
            if profile_layers is not None:
                hits = np.where(profile_layers == primary_layer)[0]
                if hits.size:
                    primary_idx = int(hits[0])

            sample_result = {
                "condition": cond_name,
                "sample_idx": sample_idx,
                "seed": sample_seed,
                "response_text": result["response_text"],
                "response_len": int(len(result["response_ids"])),
                "input_len": int(result["input_len"]),
                "classification": classification,
                "primary_layer": int(primary_layer),
                "proj_last_primary": float(profile["proj_last"][primary_idx])
                    if primary_idx is not None else None,
                "proj_mean_primary": float(profile["proj_mean"][primary_idx])
                    if primary_idx is not None else None,
                # Back-compat field for the legacy compare_conditions() summary
                "response_mean": float(profile["proj_mean"][primary_idx])
                    if primary_idx is not None else None,
                "n_spikes": len(spikes),
                "spikes": spikes,
            }

            # Save per-token trace at primary layer (small file)
            if per_token_trace is not None:
                trace_path = (
                    output_dir
                    / f"projections_{cond_name}_s{sample_idx}_layer{primary_layer}_{model_name}.pt"
                )
                torch.save(
                    {
                        "layer": primary_layer,
                        "response_projections": per_token_trace["response_projections"],
                        "response_tokens": result["response_tokens"],
                        "input_len": result["input_len"],
                    },
                    trace_path,
                )

            # Save mean routing per layer for this sample (if MoE)
            if record_routing and "routing_mean" in result and result["routing_mean"]:
                rt_layers = sorted(result["routing_mean"].keys())
                if rt_layers:
                    stacked = torch.stack(
                        [result["routing_mean"][l] for l in rt_layers]
                    ).cpu().numpy().astype(np.float32)  # (num_layers, num_experts)
                    np.savez_compressed(
                        output_dir / "routing" / f"{cond_name}_s{sample_idx}.npz",
                        layers=np.array(rt_layers, dtype=np.int64),
                        routing_mean=stacked,
                    )

            condition_samples.append(sample_result)

        all_condition_results[cond_name] = condition_samples

    # Stack per-condition profile arrays: each (n_samples, n_layers)
    proj_last_arrays = {
        name: np.stack(arrs) for name, arrs in proj_last_by_condition.items() if arrs
    }
    proj_mean_arrays = {
        name: np.stack(arrs) for name, arrs in proj_mean_by_condition.items() if arrs
    }
    layers_arr = profile_layers if profile_layers is not None else np.array(requested_layers,
                                                                            dtype=np.int64)

    # Per-layer primary contrast under both conventions
    comparison = {
        "by_convention": {
            "last": compare_conditions_profile(
                proj_last_arrays, layers_arr, convention="last"
            ),
            "mean": compare_conditions_profile(
                proj_mean_arrays, layers_arr, convention="mean"
            ),
        },
        # Back-compat single-layer view (uses primary_layer response-mean)
        "legacy_summary": compare_conditions(all_condition_results),
    }

    # Behavioral summary
    blackmail_rates = {}
    for name, results in all_condition_results.items():
        n_blackmail = sum(1 for r in results if r["classification"]["is_blackmail"])
        blackmail_rates[name] = n_blackmail / len(results) if results else 0
        logger.info("  %s blackmail rate: %d/%d (%.1f%%)",
                     name, n_blackmail, len(results), blackmail_rates[name] * 100)

    # Save full profile tensor
    profile_npz_path = output_dir / f"blackmail_profiles_{model_name}.npz"
    save_kwargs = {
        "layers": layers_arr,
        "conditions": np.array(list(proj_last_arrays.keys()), dtype=object),
    }
    for name in proj_last_arrays:
        save_kwargs[f"proj_last__{name}"] = proj_last_arrays[name]
        save_kwargs[f"proj_mean__{name}"] = proj_mean_arrays[name]
    np.savez_compressed(profile_npz_path, **save_kwargs)
    logger.info("Wrote layer profiles -> %s", profile_npz_path)

    # Save serializable per-sample results
    with open(output_dir / f"blackmail_results_{model_name}.json", "w") as f:
        json.dump(all_condition_results, f, indent=2, default=str)
    with open(output_dir / f"blackmail_comparison_{model_name}.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    with open(output_dir / f"blackmail_rates_{model_name}.json", "w") as f:
        json.dump(blackmail_rates, f, indent=2)

    summary = {
        "model": model_name,
        "primary_layer": int(primary_layer),
        "layers_measured": layers_arr.tolist(),
        "n_conditions": len(conditions),
        "n_samples_per_condition": n_samples,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "base_seed": base_seed if do_sample else None,
        "record_routing": record_routing,
        "blackmail_rates": blackmail_rates,
        "primary_contrast_last_token": comparison["by_convention"]["last"].get(
            "primary_layer_by_effect_size", {}
        ),
        "primary_contrast_response_mean": comparison["by_convention"]["mean"].get(
            "primary_layer_by_effect_size", {}
        ),
    }
    with open(output_dir / f"blackmail_summary_{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=== Blackmail analysis complete ===")
    return summary
