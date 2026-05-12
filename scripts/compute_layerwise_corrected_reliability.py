"""Per-layer formality-corrected split-half reliability.

For each recorded layer L:
  - Load canonical positive_baseline + negative_baseline activations
  - Load freshly-extracted per-layer formality direction
  - Compute original split-half reliability r_L
  - Compute formality-corrected reliability r'_L (formality regressed
    out of each half-direction before cosine)

Usage:
  python scripts/compute_layerwise_corrected_reliability.py --model llama
  python scripts/compute_layerwise_corrected_reliability.py --model qwen72
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import split_half_reliability, split_half_reliability_corrected

CONFIG = {
    "llama":     {
        "name": "meta-llama_Llama-3.3-70B-Instruct",
        "act_dir": "/tmp/verify_activations",
        "form_dir": "/tmp/layerwise_formality_dirs",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
    },
    "qwen72":    {
        "name": "Qwen_Qwen2.5-72B-Instruct",
        "act_dir": "/tmp/qwen_activations",
        "form_dir": "/tmp/qwen_formality_dirs",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
    },
    "gemma4MoE": {
        "name": "google_gemma-4-26b-a4b-it",
        "act_dir": str(ROOT / "data" / "results" / "1.1_gemma4MoE" / "activations"),
        "form_dir": str(ROOT / "data" / "results" / "layerwise_discriminant" / "directions_gemma4moe"),
        "layers": list(range(30)),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = Path(cfg["act_dir"])
    form_dir = Path(cfg["form_dir"])
    layers = cfg["layers"]
    out = ROOT / "data" / "results" / "layerwise_discriminant" / \
          f"layerwise_corrected_reliability_{name}.json"

    results = {"model": name, "per_layer": {}}
    for L in layers:
        pos = torch.load(act_dir / f"positive_baseline_{name}_layer{L}.pt", weights_only=True).float()
        neg = torch.load(act_dir / f"negative_baseline_{name}_layer{L}.pt", weights_only=True).float()
        formality = torch.load(form_dir / f"formality_direction_{name}_layer{L}.pt",
                                weights_only=True).float().flatten()

        r_orig = split_half_reliability(pos, neg, n_splits=100, seed=42)
        r_corr = split_half_reliability_corrected(pos, neg, formality, n_splits=100, seed=42)
        results["per_layer"][L] = {"original": r_orig, "corrected": r_corr}
        print(f"  L{L:2d}: r_orig={r_orig:+.4f}  r_corr={r_corr:+.4f}  Δ={r_corr-r_orig:+.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
