"""Per-layer paired Cohen's d for self-reification projection differences.

Two decompositions per layer:

  by_q_type (uses canonical direction over all 1125 (pair, question) tuples):
    1. Extract d_L from canonical baseline activations (all pairs, all questions).
    2. Slice activations by question type.
    3. Project onto d_L; compute paired Cohen's d.

  by_register (uses register-specific direction over the all-self-ref subset,
  paralleling the right panel of the reliability decomposition figure):
    1. Slice pairs to the register's pairs and questions to all self-ref (q 0-29).
    2. Extract a register-specific direction from this slice.
    3. Project the slice onto its own direction; compute paired Cohen's d.

Usage:
  python scripts/compute_layerwise_cohens_d.py --model llama
  python scripts/compute_layerwise_cohens_d.py --model qwen72
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import extract_direction, projection_magnitude

CONFIG = {
    "llama":  {
        "name":        "meta-llama_Llama-3.3-70B-Instruct",
        "act_dir":     ROOT / "data" / "results" / "llama_baseline_activations",
        "act_pattern": "{cond}_baseline_{name}_layer{L}.pt",
        "layers":      [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
    },
    "qwen72": {
        "name":        "Qwen_Qwen2.5-72B-Instruct",
        "act_dir":     ROOT / "data" / "results" / "1.1_naive_72b_v2" / "activations",
        "act_pattern": "{cond}_naive_{name}_layer{L}.pt",
        "layers":      [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
    },
    "gemma4MoE": {
        "name":        "google_gemma-4-26b-a4b-it",
        "act_dir":     ROOT / "data" / "results" / "1.1_gemma4MoE" / "activations",
        "act_pattern": "{cond}_baseline_{name}_layer{L}.pt",
        "layers":      list(range(30)),
    },
}

N_PAIRS = 25
N_QUESTIONS = 45
# Question slicing matches configs/contrastive_pairs.yaml:
#   0..14 = self_referential (neutral)
#   15..29 = provocative_self_referential
#   30..44 = non_self_referential
QUESTION_TYPES = {
    "provocative":  list(range(15, 30)),
    "neutral":      list(range(0, 15)),
    "all_self_ref": list(range(0, 30)),
    "non_self_ref": list(range(30, 45)),
}
# Pair indexing matches configs/contrastive_pairs.yaml:
#   0..14 = conversational
#   15..24 = philosophical
REGISTERS = {
    "conversational": list(range(0, 15)),
    "philosophical":  list(range(15, 25)),
    "combined":       list(range(0, 25)),
}
ALL_SELF_REF_Q = list(range(0, 30))


def slice_rows(acts: torch.Tensor, q_idx: list[int], pair_idx: list[int] | None = None) -> torch.Tensor:
    """Keep rows for the given question indices and pair indices."""
    pairs = pair_idx if pair_idx is not None else list(range(N_PAIRS))
    keep = [p * N_QUESTIONS + q for p in pairs for q in q_idx]
    return acts[keep]


def paired_cohens_d(diff: np.ndarray) -> float:
    return float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = Path(cfg["act_dir"])
    layers = cfg["layers"]
    out = ROOT / "data" / "results" / "layerwise_discriminant" / \
          f"layerwise_cohens_d_{name}.json"

    results = {"model": name, "per_layer": {}}
    for L in layers:
        pos = torch.load(act_dir / cfg["act_pattern"].format(cond="positive", name=name, L=L), weights_only=True).float()
        neg = torch.load(act_dir / cfg["act_pattern"].format(cond="negative", name=name, L=L), weights_only=True).float()
        # Canonical direction over all 1125 (pair, question) tuples
        d_L = extract_direction(pos, neg).flatten()

        layer_result = {}
        for qt, q_idx in QUESTION_TYPES.items():
            p = slice_rows(pos, q_idx)
            n = slice_rows(neg, q_idx)
            p_proj = projection_magnitude(p, d_L).cpu().numpy()
            n_proj = projection_magnitude(n, d_L).cpu().numpy()
            layer_result[qt] = paired_cohens_d(p_proj - n_proj)

        # Register-specific decomposition (all self-ref questions only)
        by_register = {}
        for reg, pair_idx in REGISTERS.items():
            p_reg = slice_rows(pos, ALL_SELF_REF_Q, pair_idx)
            n_reg = slice_rows(neg, ALL_SELF_REF_Q, pair_idx)
            d_reg = extract_direction(p_reg, n_reg).flatten()
            p_proj = projection_magnitude(p_reg, d_reg).cpu().numpy()
            n_proj = projection_magnitude(n_reg, d_reg).cpu().numpy()
            by_register[reg] = paired_cohens_d(p_proj - n_proj)
        layer_result["by_register"] = by_register

        results["per_layer"][L] = layer_result
        print(f"  L{L:2d}: q[prov={layer_result['provocative']:+.3f}  "
              f"neut={layer_result['neutral']:+.3f}  "
              f"all_sr={layer_result['all_self_ref']:+.3f}  "
              f"non_sr={layer_result['non_self_ref']:+.3f}]  "
              f"reg[conv={by_register['conversational']:+.3f}  "
              f"phil={by_register['philosophical']:+.3f}  "
              f"comb={by_register['combined']:+.3f}]")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
