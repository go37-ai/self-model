"""Per-layer formality-corrected split-half reliability — self-ref-only variant.

Same as compute_layerwise_corrected_reliability.py but slices the canonical
1125-sample (25 pairs × 45 questions) activations down to the 750-sample
self-ref-only subset (25 pairs × 30 self-ref questions: neutral + provocative)
before computing reliability. This makes the values comparable to the paper's
headline reliabilities (r=0.93 Llama, r=0.71 Qwen) which are also self-ref-only.

Usage:
    python scripts/compute_layerwise_corrected_reliability_selfref.py --model llama
    python scripts/compute_layerwise_corrected_reliability_selfref.py --model qwen
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import split_half_reliability, split_half_reliability_corrected

# Question layout: 0-14 neutral, 15-29 provocative, 30-44 non-self-ref
# Self-ref-only = first 30 of each pair
N_QUESTIONS = 45
N_PAIRS = 25
SELFREF_Q_INDICES = list(range(30))


CONFIG = {
    "llama":  {
        "name": "meta-llama_Llama-3.3-70B-Instruct",
        "act_dir": ROOT / "data" / "results" / "llama_baseline_activations",
        "form_dir": ROOT / "data" / "results" / "llama_formality_dirs",
        "act_pattern": "{cond}_baseline_{name}_layer{L}.pt",
    },
    "qwen": {
        "name": "Qwen_Qwen2.5-72B-Instruct",
        "act_dir": ROOT / "data" / "results" / "1.1_naive_72b_v2" / "activations",
        "form_dir": ROOT / "data" / "results" / "qwen_formality_dirs",
        "act_pattern": "{cond}_naive_{name}_layer{L}.pt",
    },
}
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]


def slice_to_selfref(acts):
    """Slice (25*45, hidden) -> (25*30, hidden) keeping self-ref questions only."""
    acts = acts.view(N_PAIRS, N_QUESTIONS, -1)[:, SELFREF_Q_INDICES, :]
    return acts.reshape(-1, acts.shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    name = cfg["name"]
    act_dir = cfg["act_dir"]
    form_dir = cfg["form_dir"]
    out = ROOT / "data" / "results" / "layerwise_discriminant" / \
          f"layerwise_corrected_reliability_{name}.json"

    results = {
        "model": name,
        "question_subset": "self_ref_only_30q",
        "note": "Reliability computed on 25 pairs x 30 self-ref questions (neutral + provocative). Matches paper's headline reliability subset.",
        "per_layer": {}
    }
    for L in LAYERS:
        pos_path = act_dir / cfg["act_pattern"].format(cond="positive", name=name, L=L)
        neg_path = act_dir / cfg["act_pattern"].format(cond="negative", name=name, L=L)
        form_path = form_dir / f"formality_direction_{name}_layer{L}.pt"

        pos = torch.load(pos_path, weights_only=True).float()
        neg = torch.load(neg_path, weights_only=True).float()
        formality = torch.load(form_path, weights_only=True).float().flatten()

        # Slice to self-ref-only
        pos_sr = slice_to_selfref(pos)
        neg_sr = slice_to_selfref(neg)

        r_orig = split_half_reliability(pos_sr, neg_sr, n_splits=100, seed=42)
        r_corr = split_half_reliability_corrected(pos_sr, neg_sr, formality, n_splits=100, seed=42)
        results["per_layer"][L] = {"original": r_orig, "corrected": r_corr}
        print(f"  L{L:2d}: r_orig={r_orig:+.4f}  r_corr={r_corr:+.4f}  Δ={r_corr-r_orig:+.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
