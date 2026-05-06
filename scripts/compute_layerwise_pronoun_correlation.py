"""Per-layer pronoun-density correlation with self-reification projections.

For each recorded layer L of Llama 3.3-70B:
  1. Extract the self-reification direction d_L from canonical baseline activations
  2. Project the activations of the 303 stored response samples onto d_L
  3. Compute pronoun density of each response from its text
  4. Pearson correlation between projections and densities

The 303-response file (s3://.../responses/responses_meta-llama_Llama-3.3-70B-Instruct.jsonl)
is a stratified subset of the canonical 2250-response extraction (covering all 25
pairs, both conditions, all 3 question types, both registers). The original
Table 2 number (-0.27) was computed on the full set; this analysis uses the
303-sample subset and is reported as such.

Inputs:
  /tmp/verify_activations/{positive,negative}_baseline_meta-llama_Llama-3.3-70B-Instruct_layer{L}.pt
  /tmp/llama_responses.jsonl
  configs/contrastive_pairs.yaml

Output:
  data/results/layerwise_discriminant/layerwise_pronoun_correlation_meta-llama_Llama-3.3-70B-Instruct.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import extract_direction, projection_magnitude

ACT_DIR = Path("/tmp/verify_activations")
RESP_PATH = Path("/tmp/llama_responses.jsonl")
OUT_DIR = ROOT / "data" / "results" / "layerwise_discriminant"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]

PRONOUN_RE = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)


def pronoun_density(text: str) -> float:
    """Fraction of whitespace-split tokens that are first-person pronouns."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for w in tokens if PRONOUN_RE.fullmatch(w.strip(".,!?;:'\"()[]"))) / len(tokens)


def build_question_index(config_path: Path) -> dict[str, int]:
    """Map question text to its index in the canonical (15+15+15) question list."""
    cfg = yaml.safe_load(config_path.read_text())
    eq = cfg["evaluation_questions"]
    questions = eq["self_referential"] + eq["provocative_self_referential"] + eq["non_self_referential"]
    return {q: i for i, q in enumerate(questions)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_idx = build_question_index(ROOT / "configs" / "contrastive_pairs.yaml")

    # Load 303 responses, compute pronoun densities, map to (condition, row_idx)
    samples = []  # list of dicts: condition, row_idx, density
    skipped = 0
    for line in RESP_PATH.read_text().splitlines():
        r = json.loads(line)
        q = r["question"]
        if q not in q_idx:
            skipped += 1
            continue
        row = r["pair_idx"] * 45 + q_idx[q]
        samples.append({
            "condition": r["condition"],  # "positive" or "negative"
            "row_idx": row,
            "density": pronoun_density(r["response"]),
        })
    print(f"Loaded {len(samples)} matched samples ({skipped} skipped)")

    # Per-layer correlation
    results = {"model": MODEL, "n_samples": len(samples), "per_layer": {}}
    for L in LAYERS:
        pos = torch.load(ACT_DIR / f"positive_baseline_{MODEL}_layer{L}.pt", weights_only=True).float()
        neg = torch.load(ACT_DIR / f"negative_baseline_{MODEL}_layer{L}.pt", weights_only=True).float()
        d_L = extract_direction(pos, neg).flatten()

        projections = []
        densities = []
        for s in samples:
            acts = pos if s["condition"] == "positive" else neg
            projections.append(projection_magnitude(acts[s["row_idx"]], d_L).item())
            densities.append(s["density"])

        r, p = pearsonr(projections, densities)
        results["per_layer"][L] = {"pearson_r": float(r), "p_value": float(p)}
        print(f"  L{L:2d}: r = {r:+.4f}  (p = {p:.2e})")

    out = OUT_DIR / f"layerwise_pronoun_correlation_{MODEL}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
