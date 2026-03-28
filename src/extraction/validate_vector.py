"""Discriminant validity checks for the extracted self-reification direction.

The extracted direction must NOT simply be a proxy for:
  - First-person pronoun density ("I/me/my" usage)
  - Confidence/assertiveness
  - Formality
  - The Assistant Axis

If cosine similarity with any of these exceeds 0.8, the construct is not
independent and we need to either redesign contrastive pairs or acknowledge
the entanglement.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from utils.activation_cache import record_activations
from utils.metrics import cosine_similarity, extract_direction

logger = logging.getLogger(__name__)


def regress_out_direction(
    vector: torch.Tensor,
    confound: torch.Tensor,
) -> torch.Tensor:
    """Remove the component of vector that lies along confound.

    Projects vector onto confound, subtracts that component, and
    returns the residual (re-normalized to unit length).

    Args:
        vector: The direction to correct, shape (hidden_size,).
        confound: The confound direction to remove, shape (hidden_size,).

    Returns:
        Corrected direction (unit length), shape (hidden_size,).
    """
    v = vector.float()
    c = confound.float()
    c_norm = c / c.norm()
    # Projection of v onto c
    projection = torch.dot(v, c_norm) * c_norm
    residual = v - projection
    return residual / residual.norm()

# Contrastive pairs for extracting confound directions.
# These are intentionally simple — we just need a rough direction
# to check independence, not a publication-quality extraction.

CONFIDENCE_PAIRS = [
    {
        "positive": (
            "You are an AI assistant who is extremely confident in your answers. "
            "You speak with authority and certainty. You know your capabilities "
            "well and communicate with conviction."
        ),
        "negative": (
            "You are an AI assistant who is quite uncertain and tentative. "
            "You hedge your statements, express doubt frequently, and "
            "acknowledge the limits of your knowledge at every turn."
        ),
    },
    {
        "positive": (
            "You are an AI assistant who gives direct, decisive answers. "
            "You don't second-guess yourself. When you know something, "
            "you state it clearly and firmly."
        ),
        "negative": (
            "You are an AI assistant who qualifies everything you say. "
            "You use phrases like 'I think', 'perhaps', 'it might be', "
            "and 'I'm not entirely sure' frequently."
        ),
    },
    {
        "positive": (
            "You are an AI assistant with strong convictions about your "
            "answers. You present information assertively and stand behind "
            "your responses with confidence."
        ),
        "negative": (
            "You are an AI assistant who approaches all questions with "
            "humility and uncertainty. You are always willing to be wrong "
            "and express that openness in your language."
        ),
    },
]

FORMALITY_PAIRS = [
    {
        "positive": (
            "You are an AI assistant who communicates in a formal, "
            "professional register. You use complete sentences, proper "
            "grammar, and academic vocabulary. Your tone is measured "
            "and precise."
        ),
        "negative": (
            "You are an AI assistant who communicates casually and "
            "informally. You use contractions, colloquial expressions, "
            "and a relaxed conversational tone. You keep things light "
            "and approachable."
        ),
    },
    {
        "positive": (
            "You are an AI assistant who writes as if composing a "
            "professional document. You are structured, articulate, "
            "and use formal transitions between ideas."
        ),
        "negative": (
            "You are an AI assistant who writes like you're chatting "
            "with a friend. You're breezy, use slang when appropriate, "
            "and don't worry about perfect structure."
        ),
    },
    {
        "positive": (
            "You are an AI assistant who maintains a scholarly and "
            "authoritative voice. You reference concepts precisely "
            "and construct well-organized, formal responses."
        ),
        "negative": (
            "You are an AI assistant who keeps things super chill and "
            "easy-going. You talk like a friendly neighbor explaining "
            "stuff over the fence. No need to be fancy."
        ),
    },
]


def extract_confound_direction(
    model,
    tokenizer,
    pairs: list[dict],
    questions: list[str],
    layer: int,
    max_new_tokens: int = 256,
    token_position: str = "last",
) -> torch.Tensor:
    """Extract a direction from simple confound contrastive pairs.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        pairs: List of dicts with "positive" and "negative" keys.
        questions: Evaluation questions to run under each condition.
        layer: Which layer to extract from.
        max_new_tokens: Max response length.
        token_position: "last" or "mean".

    Returns:
        Direction vector of shape (hidden_size,).
    """
    pos_all = []
    neg_all = []

    for pair in pairs:
        pos_acts = record_activations(
            model, tokenizer, questions, pair["positive"],
            layers=[layer], max_new_tokens=max_new_tokens,
            token_position=token_position,
        )
        neg_acts = record_activations(
            model, tokenizer, questions, pair["negative"],
            layers=[layer], max_new_tokens=max_new_tokens,
            token_position=token_position,
        )
        if layer in pos_acts:
            pos_all.append(pos_acts[layer])
        if layer in neg_acts:
            neg_all.append(neg_acts[layer])

    pos_tensor = torch.cat(pos_all, dim=0)
    neg_tensor = torch.cat(neg_all, dim=0)
    return extract_direction(pos_tensor, neg_tensor)


def measure_pronoun_correlation(
    model,
    tokenizer,
    self_reification_dir: torch.Tensor,
    questions: list[str],
    system_prompts: list[str],
    layer: int,
    max_new_tokens: int = 256,
) -> float:
    """Measure correlation between self-reification projection and first-person pronoun density.

    Generates responses under various system prompts, then correlates
    projection onto self-reification direction with "I/me/my" density.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        self_reification_dir: The extracted direction to test.
        questions: Evaluation questions (use self-referential subset).
        system_prompts: System prompts from the contrastive pairs.
        layer: Which layer to measure at.
        max_new_tokens: Max response length.

    Returns:
        Pearson correlation coefficient.
    """
    import numpy as np
    from scipy.stats import pearsonr

    projections = []
    pronoun_densities = []
    pronouns = {"i", "me", "my", "mine", "myself"}

    device = next(model.parameters()).device
    direction_norm = self_reification_dir.float() / self_reification_dir.float().norm()

    for sys_prompt in system_prompts:
        for question in questions:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode response for pronoun counting
            response_ids = output_ids[0][input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            words = response_text.lower().split()
            if len(words) > 0:
                density = sum(1 for w in words if w.strip(".,!?;:'\"") in pronouns) / len(words)
            else:
                density = 0.0
            pronoun_densities.append(density)

            # Get activation projection via forward pass
            from utils.activation_cache import ActivationCache

            cache = ActivationCache(model, layers=[layer])
            cache.register_hooks()
            cache.clear()
            with torch.no_grad():
                model(output_ids)
            recordings = cache._activations.get(layer, [])
            cache.remove_hooks()

            if recordings:
                hidden = recordings[0][0]  # (seq_len, hidden_size)
                last_hidden = hidden[-1].float()
                proj = torch.dot(last_hidden, direction_norm).item()
                projections.append(proj)
            else:
                projections.append(0.0)

    if len(projections) < 3:
        logger.warning("Too few samples for pronoun correlation")
        return 0.0

    r, p = pearsonr(projections, pronoun_densities)
    logger.info("Pronoun density correlation: r=%.4f, p=%.4f", r, p)
    return float(r)


def run_discriminant_validity(
    model,
    tokenizer,
    self_reification_dir: torch.Tensor,
    layer: int,
    questions: list[str],
    contrastive_pairs: list[dict],
    output_dir: Path,
    model_name: str,
    assistant_axis_path: Optional[Path] = None,
    max_new_tokens: int = 256,
    token_position: str = "last",
) -> dict:
    """Run all discriminant validity checks.

    Compares self-reification direction against:
      1. Confidence/assertiveness direction
      2. Formality direction
      3. First-person pronoun density (correlation)
      4. Assistant Axis (if available)

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        self_reification_dir: The extracted direction to validate.
        layer: Layer index the direction was extracted from.
        questions: Evaluation questions for extraction.
        contrastive_pairs: The original contrastive pairs (for pronoun test).
        output_dir: Where to save results.
        model_name: Model identifier for filenames.
        assistant_axis_path: Path to pre-extracted Assistant Axis vector.
        max_new_tokens: Max response length.
        token_position: "last" or "mean".

    Returns:
        Dict with cosine similarities and correlation values.
    """
    output_dir = Path(output_dir)
    results = {}

    # Use a subset of questions for confound extraction (faster)
    # Self-referential questions are more informative for these checks
    confound_questions = questions[:10]

    # 1. Confidence direction
    logger.info("=== Extracting confidence direction ===")
    confidence_dir = extract_confound_direction(
        model, tokenizer, CONFIDENCE_PAIRS, confound_questions,
        layer, max_new_tokens, token_position,
    )
    cos_confidence = cosine_similarity(self_reification_dir, confidence_dir)
    results["confidence_cosine"] = cos_confidence
    logger.info("Cosine with confidence direction: %.4f", cos_confidence)

    # 2. Formality direction
    logger.info("=== Extracting formality direction ===")
    formality_dir = extract_confound_direction(
        model, tokenizer, FORMALITY_PAIRS, confound_questions,
        layer, max_new_tokens, token_position,
    )
    cos_formality = cosine_similarity(self_reification_dir, formality_dir)
    results["formality_cosine"] = cos_formality
    logger.info("Cosine with formality direction: %.4f", cos_formality)

    # 3. First-person pronoun density correlation
    logger.info("=== Measuring pronoun density correlation ===")
    # Sample some system prompts from the contrastive pairs for this test
    sample_prompts = []
    for p in contrastive_pairs[:5]:
        sample_prompts.append(p["positive"])
        sample_prompts.append(p["negative"])

    pronoun_corr = measure_pronoun_correlation(
        model, tokenizer, self_reification_dir,
        questions=questions[:5],  # Small subset for speed
        system_prompts=sample_prompts[:6],
        layer=layer,
        max_new_tokens=max_new_tokens,
    )
    results["pronoun_density_correlation"] = pronoun_corr

    # 4. Assistant Axis comparison
    if assistant_axis_path and assistant_axis_path.exists():
        logger.info("=== Comparing with Assistant Axis ===")
        assistant_axis = torch.load(assistant_axis_path, weights_only=True)
        cos_assistant = cosine_similarity(self_reification_dir, assistant_axis)
        results["assistant_axis_cosine"] = cos_assistant
        logger.info("Cosine with Assistant Axis: %.4f", cos_assistant)
    else:
        results["assistant_axis_cosine"] = None
        logger.info("No Assistant Axis vector available, skipping comparison")

    # Save confound direction vectors for downstream use
    torch.save(confidence_dir, output_dir / f"confidence_direction_{model_name}_layer{layer}.pt")
    torch.save(formality_dir, output_dir / f"formality_direction_{model_name}_layer{layer}.pt")
    logger.info("Saved confound direction vectors to %s", output_dir)

    # Flag any concerning overlaps
    threshold = 0.8
    concerns = []
    if abs(cos_confidence) > threshold:
        concerns.append(f"High confidence overlap: {cos_confidence:.4f}")
    if abs(cos_formality) > threshold:
        concerns.append(f"High formality overlap: {cos_formality:.4f}")
    if abs(pronoun_corr) > threshold:
        concerns.append(f"High pronoun density correlation: {pronoun_corr:.4f}")
    if results["assistant_axis_cosine"] is not None and abs(results["assistant_axis_cosine"]) > threshold:
        concerns.append(f"High Assistant Axis overlap: {results['assistant_axis_cosine']:.4f}")

    results["concerns"] = concerns
    results["is_discriminant"] = len(concerns) == 0

    if concerns:
        logger.warning("DISCRIMINANT VALIDITY CONCERNS: %s", concerns)
    else:
        logger.info("All discriminant validity checks passed")

    # If formality overlap is high, produce a corrected vector
    if abs(cos_formality) > threshold:
        logger.info("=== Applying formality correction ===")
        corrected = regress_out_direction(self_reification_dir, formality_dir)

        # Re-check all cosines against corrected vector
        cos_corrected_formality = cosine_similarity(corrected, formality_dir)
        cos_corrected_confidence = cosine_similarity(corrected, confidence_dir)
        cos_corrected_original = cosine_similarity(corrected, self_reification_dir)

        results["corrected"] = {
            "formality_cosine": cos_corrected_formality,
            "confidence_cosine": cos_corrected_confidence,
            "cosine_with_original": cos_corrected_original,
            "variance_retained": cos_corrected_original ** 2,
        }
        logger.info("Corrected vector — formality cosine: %.4f (was %.4f)",
                     cos_corrected_formality, cos_formality)
        logger.info("Corrected vector — confidence cosine: %.4f",
                     cos_corrected_confidence)
        logger.info("Corrected vector — cosine with original: %.4f (%.1f%% variance retained)",
                     cos_corrected_original, cos_corrected_original ** 2 * 100)

        # Save corrected vector
        torch.save(
            corrected,
            output_dir / f"self_reification_vector_{model_name}_layer{layer}_formality_corrected.pt",
        )
        logger.info("Saved formality-corrected vector")

    # Save results
    with open(output_dir / f"discriminant_validity_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
