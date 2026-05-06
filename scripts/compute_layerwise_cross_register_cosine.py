"""Per-layer cross-register cosine between conversational and philosophical
self-reification directions.

For each recorded layer L:
  - dir_conv_L = extract_direction over conversational pairs (0..14)
  - dir_phil_L = extract_direction over philosophical pairs (15..24)
  - cosine(dir_conv_L, dir_phil_L)

Tests whether the cross-register unification (Llama, cos = 0.82 at L20) and
register-dependence (Qwen, cos = -0.01 at L60) hold across the whole network
or are best-layer artifacts.

Usage:
  python scripts/compute_layerwise_cross_register_cosine.py --model llama
  python scripts/compute_layerwise_cross_register_cosine.py --model qwen72
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import cosine_similarity, extract_direction

CONFIG = {
    "llama":  {"name": "meta-llama_Llama-3.3-70B-Instruct", "act_dir": "/tmp/verify_activations"},
    "qwen72": {"name": "Qwen_Qwen2.5-72B-Instruct",         "act_dir": "/tmp/qwen_activations"},
}
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]

N_PAIRS = 25
N_CONV = 15      # pairs 0..14 = conversational
N_QUESTIONS = 45


def slice_pairs(acts: torch.Tensor, pair_idxs: list[int]) -> torch.Tensor:
    keep = [p * N_QUESTIONS + q for p in pair_idxs for q in range(N_QUESTIONS)]
    return acts[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = Path(cfg["act_dir"])
    out = ROOT / "data" / "results" / "layerwise_discriminant" / \
          f"layerwise_cross_register_cosine_{name}.json"

    pair_conv = list(range(0, N_CONV))
    pair_phil = list(range(N_CONV, N_PAIRS))

    results = {"model": name, "per_layer": {}}
    for L in LAYERS:
        pos = torch.load(act_dir / f"positive_baseline_{name}_layer{L}.pt", weights_only=True).float()
        neg = torch.load(act_dir / f"negative_baseline_{name}_layer{L}.pt", weights_only=True).float()
        dir_conv = extract_direction(slice_pairs(pos, pair_conv), slice_pairs(neg, pair_conv))
        dir_phil = extract_direction(slice_pairs(pos, pair_phil), slice_pairs(neg, pair_phil))
        c = cosine_similarity(dir_conv, dir_phil)
        results["per_layer"][L] = c
        print(f"  L{L:2d}: cos(conv, phil) = {c:+.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
