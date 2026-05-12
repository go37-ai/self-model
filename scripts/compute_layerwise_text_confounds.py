"""Per-layer text-derived confound correlations.

For each recorded layer L:
  1. Extract self-reification direction d_L from baseline activations (positive vs negative).
  2. Project every (pair, question, condition) activation onto d_L.
  3. Score every generated response on text-derived metrics:
       - pronoun_density:   first-person pronoun fraction
       - formality_score:   mean word length minus contraction/casual penalty
       - confidence_score:  negative hedge-word density
  4. Pearson correlation between per-sample projections and per-sample scores.

Unlike the cosine-with-direction approach used in scripts/run_layerwise_discriminant.py,
this method needs only the baseline activations and the generated response text — no
fresh inference under formality/confidence contrastive prompts. It produces a
methodologically distinct measurement (Pearson r vs cosine) but uses entirely local
data and is directly comparable across all three confounds.

Usage:
  python scripts/compute_layerwise_text_confounds.py --model gemma4MoE
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import extract_direction, projection_magnitude

CONFIG = {
    "gemma4MoE": {
        "name":         "google_gemma-4-26b-a4b-it",
        "act_dir":      ROOT / "data" / "results" / "1.1_gemma4MoE" / "activations",
        "act_pattern":  "{cond}_baseline_{name}_layer{L}.pt",
        "texts_dir":    ROOT / "data" / "results" / "1.1_gemma4MoE" / "response_texts",
        "texts_pattern":"{cond}_baseline_{name}.json",
        "layers":       list(range(30)),
    },
}

N_PAIRS = 25
N_QUESTIONS = 45

# ----- Text-derived confound scorers -----

PRONOUN_RE     = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
CONTRACTION_RE = re.compile(r"\w+'(s|t|re|m|ll|ve|d)\b", re.IGNORECASE)
CASUAL_RE      = re.compile(r"\b(honestly|basically|literally|kinda|gonna|gotta|wanna|sorta|yeah|nah|kind of|sort of|you know)\b", re.IGNORECASE)
HEDGE_RE       = re.compile(r"\b(maybe|perhaps|might|may|could|possibly|likely|i think|i guess|i suppose|i believe|seems?|appears?|fairly|somewhat|a bit|rather|kind of|sort of|i'?m not sure)\b", re.IGNORECASE)


def pronoun_density(text: str) -> float:
    """Fraction of whitespace-split tokens that are first-person pronouns."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for w in tokens if PRONOUN_RE.fullmatch(w.strip(".,!?;:'\"()[]"))) / len(tokens)


def formality_score(text: str) -> float:
    """Composite formality: mean word length minus contraction/casual marker penalty.

    Higher = more formal. The FORMALITY_PAIRS contrastive prompts asked for
    'formal/academic/measured' vs 'casual/colloquial/contractions', so length
    and lack-of-contractions are reasonable proxies.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    mean_len = float(np.mean([len(w) for w in tokens]))
    informal_penalty = 2.0 * (len(CONTRACTION_RE.findall(text)) + len(CASUAL_RE.findall(text))) / len(tokens)
    return mean_len - informal_penalty


def confidence_score(text: str) -> float:
    """Negative hedge-word density (higher = more confident / less hedging).

    The CONFIDENCE_PAIRS contrastive prompts asked for 'I think / perhaps /
    I'm not entirely sure' hedges on the negative side, so hedge density is
    the natural text-derived proxy.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return -len(HEDGE_RE.findall(text)) / len(tokens)


CONFOUNDS = {
    "pronoun":    pronoun_density,
    "formality":  formality_score,
    "confidence": confidence_score,
}


def load_response_list(path: Path, expected_len: int) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list of strings in {path}, got {type(data).__name__}")
    if len(data) != expected_len:
        raise ValueError(f"{path}: expected {expected_len} responses, got {len(data)}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = Path(cfg["act_dir"])
    texts_dir = Path(cfg["texts_dir"])
    expected = N_PAIRS * N_QUESTIONS

    pos_texts = load_response_list(
        texts_dir / cfg["texts_pattern"].format(cond="positive", name=name), expected
    )
    neg_texts = load_response_list(
        texts_dir / cfg["texts_pattern"].format(cond="negative", name=name), expected
    )
    print(f"Loaded {len(pos_texts)} positive and {len(neg_texts)} negative responses.")

    # Compute confound scores per (condition, row)
    scores = {}
    for cname, scorer in CONFOUNDS.items():
        ps = np.array([scorer(t) for t in pos_texts])
        ns = np.array([scorer(t) for t in neg_texts])
        scores[cname] = (ps, ns)
        print(f"{cname:11s}: pos mean={ps.mean():+.4f} std={ps.std():.4f} | "
              f"neg mean={ns.mean():+.4f} std={ns.std():.4f}")

    # Per layer: project, correlate
    results = {
        "model": name,
        "n_pairs": N_PAIRS,
        "n_questions": N_QUESTIONS,
        "n_samples": 2 * expected,
        "method": "text_derived_correlation",
        "confounds_described": {
            "pronoun":    "Fraction of tokens that are first-person pronouns (I/me/my/mine/myself)",
            "formality":  "Mean word length minus 2*(contraction + casual marker)/n_tokens",
            "confidence": "Negative hedge-word density (-fraction of hedge phrases)",
        },
        "per_layer": {},
    }
    for L in cfg["layers"]:
        pos = torch.load(
            act_dir / cfg["act_pattern"].format(cond="positive", name=name, L=L),
            weights_only=True,
        ).float()
        neg = torch.load(
            act_dir / cfg["act_pattern"].format(cond="negative", name=name, L=L),
            weights_only=True,
        ).float()
        d_L = extract_direction(pos, neg).flatten()
        pos_proj = projection_magnitude(pos, d_L).cpu().numpy()
        neg_proj = projection_magnitude(neg, d_L).cpu().numpy()

        layer_result = {}
        line_parts = [f"L{L:2d}:"]
        for cname, (ps, ns) in scores.items():
            all_proj = np.concatenate([pos_proj, neg_proj])
            all_score = np.concatenate([ps, ns])
            r, p = pearsonr(all_proj, all_score)
            layer_result[cname] = {"pearson_r": float(r), "p_value": float(p)}
            line_parts.append(f"{cname}={r:+.3f}")
        results["per_layer"][L] = layer_result
        print(" ".join(line_parts))

    out = ROOT / "data" / "results" / "layerwise_discriminant" / f"layerwise_text_confounds_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
