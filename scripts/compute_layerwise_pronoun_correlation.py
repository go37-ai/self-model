"""Per-layer pronoun-density correlation with self-reification projections.

For each recorded layer L:
  1. Extract the self-reification direction d_L from canonical baseline activations.
  2. Project per-sample activations onto d_L.
  3. Compute pronoun density of each response from its text.
  4. Pearson correlation between projections and densities.

The original Llama Table 2 number (-0.27) was computed on a 303-response stratified
subset and reported as such. This script supports multiple input formats: a stratified
JSONL where each line has {question, pair_idx, condition, response} (Llama's setup),
or a pair of flat JSON lists of all responses indexed (pair_idx * N_QUESTIONS + q_idx),
matching what the main extraction writes to data/results/.../response_texts/ (Gemma).

Usage:
  python scripts/compute_layerwise_pronoun_correlation.py --model gemma4MoE
"""

import argparse
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

N_QUESTIONS = 45

CONFIG = {
    "llama": {
        "name":       "meta-llama_Llama-3.3-70B-Instruct",
        "act_dir":    Path("/tmp/verify_activations"),
        "act_pattern":"{cond}_baseline_{name}_layer{L}.pt",
        "format":     "jsonl_stratified",
        "resp_path":  Path("/tmp/llama_responses.jsonl"),
        "layers":     [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
    },
    "gemma4MoE": {
        "name":        "google_gemma-4-26b-a4b-it",
        "act_dir":     ROOT / "data" / "results" / "1.1_gemma4MoE" / "activations",
        "act_pattern": "{cond}_baseline_{name}_layer{L}.pt",
        "format":      "flat_lists",
        "texts_dir":   ROOT / "data" / "results" / "1.1_gemma4MoE" / "response_texts",
        "texts_pattern":"{cond}_baseline_{name}.json",
        "layers":      list(range(30)),
    },
}

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


def load_samples_jsonl(path: Path, q_idx: dict[str, int]) -> list[dict]:
    """Read stratified JSONL responses; map question text to row index."""
    samples = []
    skipped = 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["question"] not in q_idx:
            skipped += 1
            continue
        samples.append({
            "condition": r["condition"],
            "row_idx":   r["pair_idx"] * N_QUESTIONS + q_idx[r["question"]],
            "density":   pronoun_density(r["response"]),
        })
    print(f"Loaded {len(samples)} stratified samples ({skipped} skipped)")
    return samples


def load_samples_flat(texts_dir: Path, texts_pattern: str, name: str) -> list[dict]:
    """Read flat per-condition response lists; row_idx == list index."""
    samples = []
    for cond in ("positive", "negative"):
        path = texts_dir / texts_pattern.format(cond=cond, name=name)
        responses = json.loads(path.read_text())
        for row_idx, resp in enumerate(responses):
            samples.append({
                "condition": cond,
                "row_idx":   row_idx,
                "density":   pronoun_density(resp),
            })
    print(f"Loaded {len(samples)} samples from {texts_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = Path(cfg["act_dir"])
    out_dir = ROOT / "data" / "results" / "layerwise_discriminant"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg["format"] == "jsonl_stratified":
        q_idx = build_question_index(ROOT / "configs" / "contrastive_pairs.yaml")
        samples = load_samples_jsonl(Path(cfg["resp_path"]), q_idx)
    else:  # flat_lists
        samples = load_samples_flat(Path(cfg["texts_dir"]), cfg["texts_pattern"], name)

    results = {"model": name, "n_samples": len(samples), "per_layer": {}}
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

        projections = []
        densities = []
        for s in samples:
            acts = pos if s["condition"] == "positive" else neg
            projections.append(projection_magnitude(acts[s["row_idx"]], d_L).item())
            densities.append(s["density"])

        r, p = pearsonr(projections, densities)
        results["per_layer"][L] = {"pearson_r": float(r), "p_value": float(p)}
        print(f"  L{L:2d}: r = {r:+.4f}  (p = {p:.2e})")

    out = out_dir / f"layerwise_pronoun_correlation_{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
