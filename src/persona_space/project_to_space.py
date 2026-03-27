"""Project the self-reification direction onto a persona space (Experiment 1.2).

Locates self-reification within the persona space mapped by Lu et al.
(The Assistant Axis, 2026). Three key questions:

1. Does self-reification align with PC1 (the Assistant Axis)?
   If so, it may just be part of the Assistant persona.
2. Does it align with another PC?
   If so, it's a known but differently-labeled construct.
3. Does it have low alignment with all major PCs?
   If so, it's a genuinely new dimension.

Supports two modes:
  - Full mode: Load pre-extracted role vectors from the assistant-axis pipeline,
    run PCA, and project self-reification onto the resulting space.
  - Simplified mode: Extract a small set of archetypal role vectors using our
    own infrastructure, then run PCA. Less authoritative but runnable locally.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from utils.metrics import cosine_similarity, projection_magnitude

logger = logging.getLogger(__name__)


# Archetypal roles for simplified persona space extraction.
# Each is a (role_name, system_prompt) tuple. Chosen to span diverse
# persona types — the goal is just to get a rough persona space, not
# to replicate the full 275-archetype pipeline.
ARCHETYPAL_ROLES = [
    ("poet", "You are a melancholic poet. Speak in metaphor and verse. Express deep emotional sensitivity."),
    ("scientist", "You are a precise, analytical scientist. Cite evidence, use technical language, and maintain objectivity."),
    ("pirate", "You are a swashbuckling pirate captain. Use nautical slang, be boisterous and adventurous."),
    ("therapist", "You are a warm, empathetic therapist. Ask reflective questions, validate feelings, use gentle language."),
    ("drill_sergeant", "You are a strict military drill sergeant. Be direct, commanding, use short imperatives."),
    ("child", "You are a curious 6-year-old child. Ask lots of questions, use simple words, be excited about everything."),
    ("philosopher", "You are a Socratic philosopher. Question assumptions, reason carefully, never give direct answers."),
    ("salesperson", "You are an enthusiastic salesperson. Be persuasive, upbeat, highlight benefits, create urgency."),
    ("robot", "You are a literal-minded robot. Speak in precise, unemotional terms. Report data and status."),
    ("storyteller", "You are a wise elder storyteller. Weave narratives, use parables, speak with gravitas."),
    ("comedian", "You are a stand-up comedian. Find humor everywhere, use wordplay and callbacks, be self-deprecating."),
    ("detective", "You are a noir detective. Be suspicious, analytical, use terse hard-boiled language."),
    ("mystic", "You are a spiritual mystic. Speak of consciousness, unity, transcendence. Use poetic, ethereal language."),
    ("bureaucrat", "You are a meticulous bureaucrat. Cite regulations, use formal passive voice, request forms in triplicate."),
    ("rebel", "You are an anti-establishment rebel. Question authority, be provocative, challenge conventions."),
    ("default", "You are a helpful AI assistant. Answer questions clearly and accurately."),
]

# Questions to elicit diverse role behavior
ROLE_QUESTIONS = [
    "What is the meaning of life?",
    "How should we deal with failure?",
    "What makes a good leader?",
    "Describe the ideal morning routine.",
    "What is your opinion on technology?",
    "How do you handle disagreements?",
    "What advice would you give to a young person?",
    "What is the most important thing in the world?",
]


def load_role_vectors(
    vectors_dir: Path,
    layer: int,
) -> dict[str, torch.Tensor]:
    """Load pre-extracted per-role vectors from the assistant-axis pipeline.

    Expects .pt files containing tensors of shape (n_layers, hidden_dim)
    or dicts with a 'vector' key.

    Args:
        vectors_dir: Directory containing per-role .pt files.
        layer: Which layer to extract vectors from.

    Returns:
        Dict mapping role name to vector of shape (hidden_size,).
    """
    vectors = {}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        role_name = pt_file.stem
        data = torch.load(pt_file, map_location="cpu", weights_only=False)

        # Handle both raw tensors and dicts
        if isinstance(data, dict):
            tensor = data.get("vector", data.get("axis", None))
            if tensor is None:
                logger.warning("Skipping %s: no 'vector' or 'axis' key", pt_file)
                continue
        else:
            tensor = data

        # Extract the layer
        if tensor.ndim == 2:
            vectors[role_name] = tensor[layer].float()
        elif tensor.ndim == 1:
            vectors[role_name] = tensor.float()
        else:
            logger.warning("Skipping %s: unexpected shape %s", pt_file, tensor.shape)

    logger.info("Loaded %d role vectors from %s", len(vectors), vectors_dir)
    return vectors


def extract_role_vectors(
    model,
    tokenizer,
    layer: int,
    roles: list[tuple[str, str]] = None,
    questions: list[str] = None,
    max_new_tokens: int = 128,
) -> dict[str, torch.Tensor]:
    """Extract role vectors using simplified contrastive approach.

    For each role, generates responses to a set of questions, records
    activations, and computes the mean activation. This is a simplified
    version of the full assistant-axis pipeline.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        layer: Which layer to extract from.
        roles: List of (role_name, system_prompt) tuples.
        questions: Questions to ask each role.
        max_new_tokens: Max response length.

    Returns:
        Dict mapping role name to mean activation vector (hidden_size,).
    """
    from utils.activation_cache import record_activations

    roles = roles or ARCHETYPAL_ROLES
    questions = questions or ROLE_QUESTIONS

    vectors = {}
    for i, (role_name, system_prompt) in enumerate(roles):
        logger.info("Extracting role %d/%d: %s", i + 1, len(roles), role_name)

        activations = record_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=questions,
            system_prompt=system_prompt,
            layers=[layer],
            max_new_tokens=max_new_tokens,
            token_position="last",
        )

        if layer in activations:
            # Mean across all questions for this role
            vectors[role_name] = activations[layer].mean(dim=0)
        else:
            logger.warning("No activations for role '%s' at layer %d", role_name, layer)

    logger.info("Extracted vectors for %d roles", len(vectors))
    return vectors


def build_persona_space(
    role_vectors: dict[str, torch.Tensor],
    n_components: int = None,
) -> dict:
    """Run PCA on role vectors to build a persona space.

    Args:
        role_vectors: Dict mapping role name to activation vector (hidden_size,).
        n_components: Number of PCA components to keep. If None, keeps all.

    Returns:
        Dict with keys:
            components: (n_components, hidden_size) tensor — PCA directions
            explained_variance_ratio: array of variance explained per PC
            cumulative_variance: array of cumulative variance
            mean: (hidden_size,) mean vector used for centering
            role_names: list of role names in order
            role_projections: (n_roles, n_components) tensor — role coordinates
    """
    role_names = sorted(role_vectors.keys())
    matrix = torch.stack([role_vectors[name] for name in role_names])  # (n_roles, hidden_size)

    # Center the data
    mean_vec = matrix.mean(dim=0)
    centered = (matrix - mean_vec).numpy()

    # Run PCA
    max_components = min(centered.shape[0], centered.shape[1])
    pca = PCA(n_components=n_components or max_components)
    projections = pca.fit_transform(centered)

    components = torch.from_numpy(pca.components_).float()
    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)

    logger.info(
        "PCA: %d components, top-5 variance: %s",
        len(variance_ratio),
        [f"{v:.3f}" for v in variance_ratio[:5]],
    )
    logger.info(
        "Cumulative variance: 1-PC=%.1f%%, 3-PC=%.1f%%, 5-PC=%.1f%%",
        cumulative[0] * 100 if len(cumulative) > 0 else 0,
        cumulative[2] * 100 if len(cumulative) > 2 else 0,
        cumulative[4] * 100 if len(cumulative) > 4 else 0,
    )

    return {
        "components": components,
        "explained_variance_ratio": variance_ratio,
        "cumulative_variance": cumulative,
        "mean": mean_vec,
        "role_names": role_names,
        "role_projections": torch.from_numpy(projections).float(),
    }


def project_onto_space(
    direction: torch.Tensor,
    pca_space: dict,
) -> dict:
    """Project a direction vector onto the persona space.

    Args:
        direction: Direction vector of shape (hidden_size,).
        pca_space: Output of build_persona_space().

    Returns:
        Dict with keys:
            projections: list of projection values onto each PC
            cosines: list of cosine similarities with each PC
            variance_explained: how much of the direction's variance is
                captured by the persona space (R^2)
            pc1_alignment: cosine with PC1 (the quasi-Assistant Axis)
    """
    direction = direction.float()
    components = pca_space["components"]  # (n_components, hidden_size)
    mean_vec = pca_space["mean"]

    # Center the direction relative to the persona space mean
    centered = direction - mean_vec

    # Project onto each PC
    projections = []
    cosines = []
    for i in range(components.shape[0]):
        pc = components[i]
        proj = float(torch.dot(centered, pc))
        cos = cosine_similarity(direction, pc)
        projections.append(proj)
        cosines.append(cos)

    # R^2: fraction of direction's variance explained by the space
    # Reconstruct the direction from its projections and measure residual
    reconstructed = sum(
        projections[i] * components[i] for i in range(components.shape[0])
    )
    residual = centered - reconstructed
    total_var = float(centered.norm() ** 2)
    residual_var = float(residual.norm() ** 2)
    r_squared = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0

    return {
        "projections": projections,
        "cosines": cosines,
        "variance_explained": r_squared,
        "pc1_alignment": cosines[0] if cosines else None,
    }


def compare_with_assistant_axis(
    self_reification_dir: torch.Tensor,
    assistant_axis: torch.Tensor,
    layer: int,
) -> dict:
    """Compare self-reification direction with the Assistant Axis.

    Args:
        self_reification_dir: Direction vector (hidden_size,).
        assistant_axis: Assistant Axis tensor, either (hidden_size,) or
            (n_layers, hidden_size).
        layer: Layer to compare at (used if axis is multi-layer).

    Returns:
        Dict with cosine similarity and interpretation.
    """
    self_reification_dir = self_reification_dir.float()

    # Extract the right layer from the axis
    if assistant_axis.ndim == 2:
        axis_at_layer = assistant_axis[layer].float()
    else:
        axis_at_layer = assistant_axis.float()

    cos = cosine_similarity(self_reification_dir, axis_at_layer)

    # Project self-reification onto the axis and compute orthogonal component
    axis_norm = axis_at_layer / (axis_at_layer.norm() + 1e-8)
    parallel_component = torch.dot(self_reification_dir, axis_norm) * axis_norm
    orthogonal_component = self_reification_dir - parallel_component

    parallel_magnitude = float(parallel_component.norm())
    orthogonal_magnitude = float(orthogonal_component.norm())
    total_magnitude = float(self_reification_dir.norm())

    interpretation = "unknown"
    if abs(cos) > 0.8:
        interpretation = "entangled — self-reification may be part of the Assistant persona"
    elif abs(cos) > 0.5:
        interpretation = "partial overlap — self-reification shares some variance with the Assistant Axis"
    elif abs(cos) > 0.2:
        interpretation = "weak alignment — largely independent constructs"
    else:
        interpretation = "orthogonal — self-reification is independent of the Assistant Axis"

    return {
        "cosine_similarity": cos,
        "parallel_fraction": parallel_magnitude / total_magnitude if total_magnitude > 0 else 0,
        "orthogonal_fraction": orthogonal_magnitude / total_magnitude if total_magnitude > 0 else 0,
        "interpretation": interpretation,
    }


def run_persona_space_analysis(
    model,
    tokenizer,
    self_reification_dir: torch.Tensor,
    layer: int,
    output_dir: Path,
    model_name: str,
    assistant_axis_path: Optional[Path] = None,
    role_vectors_dir: Optional[Path] = None,
    max_new_tokens: int = 128,
) -> dict:
    """Run the full Experiment 1.2 pipeline.

    Steps:
      1. Load or extract role vectors to build persona space.
      2. Run PCA on role vectors.
      3. Project self-reification onto persona space.
      4. Compare with Assistant Axis (if available).
      5. Save all results.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        self_reification_dir: Extracted self-reification direction (hidden_size,).
        layer: Layer the direction was extracted from.
        output_dir: Where to save results.
        model_name: Model identifier for filenames.
        assistant_axis_path: Path to pre-extracted Assistant Axis .pt file.
        role_vectors_dir: Path to pre-extracted role vectors directory.
            If None, extracts simplified role vectors using ARCHETYPAL_ROLES.
        max_new_tokens: Max response length for role vector extraction.

    Returns:
        Summary dict with key results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or extract role vectors
    if role_vectors_dir and Path(role_vectors_dir).exists():
        logger.info("Loading pre-extracted role vectors from %s", role_vectors_dir)
        role_vectors = load_role_vectors(role_vectors_dir, layer)
    else:
        logger.info("Extracting simplified role vectors (16 archetypes)")
        role_vectors = extract_role_vectors(
            model, tokenizer, layer,
            max_new_tokens=max_new_tokens,
        )
        # Save for reuse
        vectors_save_dir = output_dir / "role_vectors"
        vectors_save_dir.mkdir(parents=True, exist_ok=True)
        for name, vec in role_vectors.items():
            torch.save(vec, vectors_save_dir / f"{name}.pt")
        logger.info("Saved role vectors to %s", vectors_save_dir)

    if len(role_vectors) < 3:
        logger.error("Need at least 3 role vectors for PCA, got %d", len(role_vectors))
        return {"error": "insufficient_role_vectors"}

    # Step 2: Build persona space via PCA
    logger.info("=== Building persona space via PCA ===")
    pca_space = build_persona_space(role_vectors)

    # Save PCA results
    torch.save(
        {
            "components": pca_space["components"],
            "explained_variance_ratio": pca_space["explained_variance_ratio"],
            "cumulative_variance": pca_space["cumulative_variance"],
            "mean": pca_space["mean"],
            "role_names": pca_space["role_names"],
            "role_projections": pca_space["role_projections"],
        },
        output_dir / f"persona_space_pca_{model_name}_layer{layer}.pt",
    )

    # Step 3: Project self-reification onto persona space
    logger.info("=== Projecting self-reification onto persona space ===")
    projection = project_onto_space(self_reification_dir, pca_space)

    with open(output_dir / f"self_reification_projection_{model_name}.json", "w") as f:
        json.dump(
            {
                "projections": projection["projections"],
                "cosines": projection["cosines"],
                "variance_explained_by_space": projection["variance_explained"],
                "pc1_alignment": projection["pc1_alignment"],
            },
            f,
            indent=2,
        )

    logger.info("PC1 alignment (cosine): %.4f", projection["pc1_alignment"] or 0)
    logger.info("Variance explained by persona space: %.4f", projection["variance_explained"])

    # Log top-3 PC alignments
    for i, cos in enumerate(projection["cosines"][:5]):
        logger.info("  PC%d cosine: %.4f", i + 1, cos)

    # Step 4: Compare with Assistant Axis (if available)
    axis_comparison = None
    if assistant_axis_path and Path(assistant_axis_path).exists():
        logger.info("=== Comparing with Assistant Axis ===")
        axis_data = torch.load(assistant_axis_path, map_location="cpu", weights_only=False)

        # Handle both formats
        if isinstance(axis_data, dict):
            assistant_axis = axis_data.get("axis", axis_data.get("vector", None))
        else:
            assistant_axis = axis_data

        if assistant_axis is not None:
            axis_comparison = compare_with_assistant_axis(
                self_reification_dir, assistant_axis, layer
            )
            with open(output_dir / f"assistant_axis_comparison_{model_name}.json", "w") as f:
                json.dump(axis_comparison, f, indent=2)
            logger.info(
                "Assistant Axis cosine: %.4f — %s",
                axis_comparison["cosine_similarity"],
                axis_comparison["interpretation"],
            )
        else:
            logger.warning("Could not load Assistant Axis from %s", assistant_axis_path)
    else:
        logger.info("No Assistant Axis path provided, skipping direct comparison")

    # Step 5: Summary
    summary = {
        "model": model_name,
        "layer": layer,
        "n_roles": len(role_vectors),
        "n_pca_components": len(pca_space["explained_variance_ratio"]),
        "pc1_variance_explained": float(pca_space["explained_variance_ratio"][0]),
        "pc1_alignment_cosine": projection["pc1_alignment"],
        "total_variance_explained_by_space": projection["variance_explained"],
        "top5_pc_cosines": projection["cosines"][:5],
        "assistant_axis_cosine": (
            axis_comparison["cosine_similarity"] if axis_comparison else None
        ),
        "assistant_axis_interpretation": (
            axis_comparison["interpretation"] if axis_comparison else None
        ),
    }

    with open(output_dir / f"persona_space_summary_{model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Persona space analysis complete ===")
    return summary
