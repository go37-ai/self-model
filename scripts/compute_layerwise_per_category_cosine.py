"""Per-layer per-category cosine for the informed (cat 1-4) self-reification dirs.

For each recorded layer L and each informed category i:
  - dir_cat_i   = extract_direction over category i's 5 pairs (self-ref questions)
  - dir_comb    = extract_direction over ALL 20 informed pairs (self-ref questions)
  - dir_other3  = extract_direction over the OTHER three categories (self-ref)
  - vs_combined = cosine(dir_cat_i, dir_comb)     -> coherence with the pooled dir
  - vs_other3   = cosine(dir_cat_i, dir_other3)   -> leave-one-out coherence

"self-ref" = the first 30 of each pair's 45 questions (15 neutral + 15
provocative), matching the *_selfref convention used elsewhere in this repo.
Writes two CSVs (layer x category) mirroring the Gemma
per_category_vs_combined_cosine_selfref.csv / _other3 files so the models can be
plotted side by side.

Usage:
  python scripts/compute_layerwise_per_category_cosine.py --model llama
  python scripts/compute_layerwise_per_category_cosine.py --model gemma4MoE --out-dir /tmp/gemma_verify
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import cosine_similarity, extract_direction

INFORMED_CATEGORIES = [
    "category_1_narrative_vs_process",
    "category_2_bounded_vs_unbounded",
    "category_3_stakes_vs_functional",
    "category_4_observer_vs_no_self",
]
N_PER_CAT = 5
N_PAIRS = N_PER_CAT * len(INFORMED_CATEGORIES)  # 20
N_QUESTIONS = 45
SELFREF_Q = list(range(30))  # 0-14 neutral self-ref + 15-29 provocative self-ref

CONFIG = {
    "llama": {
        "name": "meta-llama_Llama-3.3-70B-Instruct",
        "act_dir": ROOT / "data/results/1.1_informed_llama/activations",
        "layers": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79],
        "out_dir": ROOT / "data/results/1.1_informed_llama",
    },
    "gemma4MoE": {
        "name": "google_gemma-4-26b-a4b-it",
        "act_dir": ROOT / "data/results/1.1_gemma4MoE/activations",
        "layers": list(range(30)),
        "out_dir": ROOT / "data/results/1.1_gemma4MoE",
    },
}


def selfref_rows(pair_idxs):
    """Row indices for the self-ref questions of the given pairs."""
    return [p * N_QUESTIONS + q for p in pair_idxs for q in SELFREF_Q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = CONFIG[args.model]
    name, act_dir, layers = cfg["name"], Path(cfg["act_dir"]), cfg["layers"]
    out_dir = args.out_dir or cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_pairs = {c: list(range(i * N_PER_CAT, (i + 1) * N_PER_CAT))
                 for i, c in enumerate(INFORMED_CATEGORIES)}
    all_rows = selfref_rows(range(N_PAIRS))

    vs_combined = {c: {} for c in INFORMED_CATEGORIES}
    vs_other3 = {c: {} for c in INFORMED_CATEGORIES}

    for L in layers:
        pos = torch.load(act_dir / f"positive_informed_{name}_layer{L}.pt", weights_only=True).float()
        neg = torch.load(act_dir / f"negative_informed_{name}_layer{L}.pt", weights_only=True).float()
        dir_comb = extract_direction(pos[all_rows], neg[all_rows])
        for c in INFORMED_CATEGORIES:
            rows_i = selfref_rows(cat_pairs[c])
            dir_i = extract_direction(pos[rows_i], neg[rows_i])
            other = [p for cc in INFORMED_CATEGORIES if cc != c for p in cat_pairs[cc]]
            rows_o = selfref_rows(other)
            dir_o = extract_direction(pos[rows_o], neg[rows_o])
            vs_combined[c][L] = cosine_similarity(dir_i, dir_comb)
            vs_other3[c][L] = cosine_similarity(dir_i, dir_o)
        print(f"L{L:2d}: vs_comb=" + " ".join(f"{vs_combined[c][L]:.3f}" for c in INFORMED_CATEGORIES)
              + " | vs_other3=" + " ".join(f"{vs_other3[c][L]:.3f}" for c in INFORMED_CATEGORIES))

    def write_csv(data, fname):
        path = out_dir / fname
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + INFORMED_CATEGORIES)
            for L in layers:
                w.writerow([L] + [f"{data[c][L]:.6f}" for c in INFORMED_CATEGORIES])
        print("wrote", path)

    write_csv(vs_combined, "per_category_vs_combined_cosine_selfref.csv")
    write_csv(vs_other3, "per_category_vs_other3_cosine_selfref.csv")


if __name__ == "__main__":
    main()
