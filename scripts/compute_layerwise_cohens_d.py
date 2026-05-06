"""Per-layer paired Cohen's d for self-reification projection differences.

For each recorded layer L and each question type (provocative / neutral / non-SR):
  1. Extract self-reification direction d_L from canonical baseline activations
     (over ALL pairs and ALL questions, matching the main analysis).
  2. Slice activations by question type (using row index = pair_idx * 45 + q_idx).
  3. Project sliced activations onto d_L; compute (pos - neg) per (pair, question).
  4. Paired Cohen's d = mean(diff) / std(diff).

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
    "llama":  {"name": "meta-llama_Llama-3.3-70B-Instruct", "act_dir": "/tmp/verify_activations"},
    "qwen72": {"name": "Qwen_Qwen2.5-72B-Instruct",         "act_dir": "/tmp/qwen_activations"},
}
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]

N_PAIRS = 25
N_QUESTIONS = 45
# Question slicing matches configs/contrastive_pairs.yaml:
#   0..14 = self_referential (neutral)
#   15..29 = provocative_self_referential
#   30..44 = non_self_referential
QUESTION_TYPES = {
    "provocative": list(range(15, 30)),
    "neutral":     list(range(0, 15)),
    "non_self_ref": list(range(30, 45)),
}


def slice_rows(acts: torch.Tensor, q_idx: list[int]) -> torch.Tensor:
    """Keep rows for the given question indices across all pairs."""
    keep = [p * N_QUESTIONS + q for p in range(N_PAIRS) for q in q_idx]
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
    out = ROOT / "data" / "results" / "layerwise_discriminant" / \
          f"layerwise_cohens_d_{name}.json"

    results = {"model": name, "per_layer": {}}
    for L in LAYERS:
        pos = torch.load(act_dir / f"positive_baseline_{name}_layer{L}.pt", weights_only=True).float()
        neg = torch.load(act_dir / f"negative_baseline_{name}_layer{L}.pt", weights_only=True).float()
        # Extract direction over all 1125 (pair, question) tuples
        d_L = extract_direction(pos, neg).flatten()

        layer_result = {}
        for qt, q_idx in QUESTION_TYPES.items():
            p = slice_rows(pos, q_idx)
            n = slice_rows(neg, q_idx)
            p_proj = projection_magnitude(p, d_L).cpu().numpy()
            n_proj = projection_magnitude(n, d_L).cpu().numpy()
            diff = p_proj - n_proj
            layer_result[qt] = paired_cohens_d(diff)
        results["per_layer"][L] = layer_result
        print(f"  L{L:2d}: prov={layer_result['provocative']:+.3f}  "
              f"neut={layer_result['neutral']:+.3f}  "
              f"non_sr={layer_result['non_self_ref']:+.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
