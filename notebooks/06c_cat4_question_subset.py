"""Item analysis on Qwen3-32B Cat 4 (observer/no-self) activations.

Goal: see whether item-pruning the question set raises the per-pair signal,
which would mean the published Cat 4 reliability of r=0.148 understates the
construct's recoverability at small N.

Activation layout (from extraction.log):
  20 informed pairs × 30 questions, where:
    pairs 0-4 = cat 1 (narrative)
    pairs 5-9 = cat 2 (bounded)
    pairs 10-14 = cat 3 (stakes)
    pairs 15-19 = cat 4 (observer)  <-- the construct of interest
  questions 0-14 = self_referential
  questions 15-29 = provocative_self_referential

Cat 4's published peak reliability is r=0.148 at L49 with all 30 questions.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/brian/repos/self-model")
ACT_DIR = REPO / "data/results/1.1_informed/activations"
MODEL = "Qwen_Qwen3-32B"
LAYER = 49  # Cat 4 peak layer
NUM_PAIRS = 20
NUM_QUESTIONS = 30
CAT4_PAIRS = list(range(15, 20))

random.seed(0)
np.random.seed(0)


def load_acts(condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_informed_{MODEL}_layer{LAYER}.pt"
    t = torch.load(p, weights_only=True, map_location="cpu").float()
    return t.view(NUM_PAIRS, NUM_QUESTIONS, -1)


pos = load_acts("positive")
neg = load_acts("negative")
print(f"Loaded: pos {tuple(pos.shape)}, neg {tuple(neg.shape)}")


def direction_from(pair_idx: list[int], q_idx: list[int]) -> torch.Tensor:
    p_sub = pos[pair_idx][:, q_idx, :].reshape(-1, pos.shape[-1])
    n_sub = neg[pair_idx][:, q_idx, :].reshape(-1, neg.shape[-1])
    return p_sub.mean(0) - n_sub.mean(0)


SUBSETS = {
    "self_ref (Q0-14)":     list(range(0, 15)),
    "provocative (Q15-29)": list(range(15, 30)),
    "all 30 (paper config)": list(range(0, 30)),
}


# 1. Pairwise cosine between subset directions (Cat 4 only, all 5 pairs)
print("\n=== Cosine between subset directions (Cat 4, all 5 pairs) ===")
dirs = {name: direction_from(CAT4_PAIRS, q) for name, q in SUBSETS.items()}
for n1 in SUBSETS:
    for n2 in SUBSETS:
        if n1 < n2:
            c = torch.nn.functional.cosine_similarity(dirs[n1], dirs[n2], dim=0).item()
            print(f"  {n1:25s} vs {n2:25s}  cos = {c:.3f}")


# 2. Split-half reliability per subset.
# At N=5 pairs, splits are 2 vs 3, which is very noisy. Use all C(5,2)=10
# possible splits and average. Also bootstrap pair sampling for stability
# (with replacement, giving slightly different N=5 sets) — though here we'd
# be sampling from a tiny universe of 5, so the bootstrap is mostly to get
# uncertainty on the deterministic 10 splits.
print("\n=== Split-half reliability for Cat 4 (N=5 pairs, all 10 splits) ===")
from itertools import combinations
print(f"  {'subset':30s}  {'r (mean ± std over splits)':>30s}")
for name, q_idx in SUBSETS.items():
    rs = []
    for half_a in combinations(CAT4_PAIRS, 2):
        half_b = [p for p in CAT4_PAIRS if p not in half_a][:2]
        a = direction_from(list(half_a), q_idx)
        b = direction_from(half_b, q_idx)
        rs.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    rs = np.array(rs)
    print(f"  {name:30s}  {rs.mean():.4f} ± {rs.std():.4f}  (n_splits={len(rs)})")


# 3. Apply Spearman-Brown to predict N=25 for each subset
def spearman_brown(r: float, k: float) -> float:
    return (k * r) / (1 + (k - 1) * r)


print("\n=== Spearman-Brown extrapolation: Cat 4 at N=25 (k=5) ===")
for name, q_idx in SUBSETS.items():
    rs = []
    for half_a in combinations(CAT4_PAIRS, 2):
        half_b = [p for p in CAT4_PAIRS if p not in half_a][:2]
        a = direction_from(list(half_a), q_idx)
        b = direction_from(half_b, q_idx)
        rs.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    r_mean = np.mean(rs)
    r5 = max(r_mean, 0.0)  # SB is undefined for negative r
    r25 = spearman_brown(r5, 5)
    print(f"  {name:30s}  r(N=5) = {r_mean:.3f}  →  predicted r(N=25) = {r25:.3f}")


# 4. Sanity check: also do the same on Cat 5 baseline (which had higher r)
# from the same 32B run. Need to load baseline activations.
print("\n=== Sanity: same analysis on Cat 5 baseline (32B) for comparison ===")
naive_dir = REPO / "data/results/1.1_naive_72b_v2/activations"
try:
    pos_b = torch.load(
        naive_dir / f"positive_naive_{MODEL}_layer63.pt",
        weights_only=True, map_location="cpu",
    ).float()
    neg_b = torch.load(
        naive_dir / f"negative_naive_{MODEL}_layer63.pt",
        weights_only=True, map_location="cpu",
    ).float()
    # baseline run had 25 pairs × 45 questions
    pos_b = pos_b.view(25, 45, -1)
    neg_b = neg_b.view(25, 45, -1)
    print(f"  Loaded baseline: {tuple(pos_b.shape)}")

    BASELINE_SUBSETS = {
        "self_ref (Q0-14)":      list(range(0, 15)),
        "provocative (Q15-29)":  list(range(15, 30)),
        "non_self_ref (Q30-44)": list(range(30, 45)),
        "all 45":                list(range(0, 45)),
    }

    def dir_b(p_idx, q_idx):
        p_sub = pos_b[p_idx][:, q_idx, :].reshape(-1, pos_b.shape[-1])
        n_sub = neg_b[p_idx][:, q_idx, :].reshape(-1, neg_b.shape[-1])
        return p_sub.mean(0) - n_sub.mean(0)

    all_pairs = list(range(25))
    print(f"  {'subset':30s}  {'r (200 random splits)':>22s}")
    for name, q_idx in BASELINE_SUBSETS.items():
        rs = []
        for _ in range(200):
            idx = random.sample(all_pairs, 25)
            a = dir_b(idx[:12], q_idx)
            b = dir_b(idx[12:24], q_idx)
            rs.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
        rs = np.array(rs)
        print(f"  {name:30s}  {rs.mean():.4f} ± {rs.std():.4f}")
except FileNotFoundError as e:
    print(f"  (baseline 32B activations not found locally: {e})")
