"""Test whether non-self-ref questions contribute signal to the extracted direction.

Question structure (45 total per pair):
  Q[0:15]   = self_referential
  Q[15:30]  = provocative_self_referential
  Q[30:45]  = non_self_referential (controls)

For each subset, compute:
  - Direction extracted using only that subset
  - Cosine similarity between subset directions (do they all measure the same thing?)
  - Split-half reliability using only that subset

If non-self-ref questions are pure controls, they should show:
  - Low cosine with self-ref directions
  - Lower reliability (more noise, less construct signal)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/brian/repos/self-model")
ACT_DIR = REPO / "data/results/llama_baseline_activations"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
LAYER = 20
NUM_PAIRS = 25
NUM_QUESTIONS = 45

random.seed(0)
np.random.seed(0)


def load_acts(condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_baseline_{MODEL}_layer{LAYER}.pt"
    t = torch.load(p, weights_only=True, map_location="cpu").float()
    return t.view(NUM_PAIRS, NUM_QUESTIONS, -1)


pos = load_acts("positive")
neg = load_acts("negative")

# Question subsets
SUBSETS = {
    "self_ref (Q0-14)":          list(range(0, 15)),
    "provocative (Q15-29)":      list(range(15, 30)),
    "non_self_ref (Q30-44)":     list(range(30, 45)),
    "all self-ref (Q0-29)":      list(range(0, 30)),
    "all 45 (paper config)":     list(range(0, 45)),
}


def direction_from(pair_idx: list[int], q_idx: list[int]) -> torch.Tensor:
    p_sub = pos[pair_idx][:, q_idx, :].reshape(-1, pos.shape[-1])
    n_sub = neg[pair_idx][:, q_idx, :].reshape(-1, neg.shape[-1])
    return p_sub.mean(0) - n_sub.mean(0)


# 1. Full direction from each question subset (all 25 pairs)
print("Direction extracted with all 25 pairs, varying question subset:")
print()
all_pairs = list(range(NUM_PAIRS))
directions = {name: direction_from(all_pairs, q) for name, q in SUBSETS.items()}

print("Pairwise cosine between subset directions:")
print(f"{'':30s}", end="")
for name in SUBSETS:
    print(f"{name[:14]:>15s}", end="")
print()
for n1 in SUBSETS:
    print(f"{n1[:30]:30s}", end="")
    for n2 in SUBSETS:
        c = torch.nn.functional.cosine_similarity(directions[n1], directions[n2], dim=0).item()
        print(f"{c:15.3f}", end="")
    print()


# 2. Split-half reliability using each question subset
print("\n\nSplit-half reliability by question subset (200 random pair-splits each):")
print(f"  {'subset':30s}  {'r (mean ± std)':>18s}")
print(f"  {'-' * 30}  {'-' * 18}")

N_TRIALS = 200
for name, q_idx in SUBSETS.items():
    rs = []
    for _ in range(N_TRIALS):
        p_idx = random.sample(all_pairs, NUM_PAIRS)
        half = NUM_PAIRS // 2
        a = direction_from(p_idx[:half], q_idx)
        b = direction_from(p_idx[half:half * 2], q_idx)
        rs.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    rs = np.array(rs)
    print(f"  {name:30s}  {rs.mean():.4f} ± {rs.std():.4f}")


# 3. Item-total correlation: does each question subset's direction align with
#    the all-questions direction? (Loose analog of item-total correlation)
print("\n\nCosine of each subset direction with the 'all 45' direction:")
all45 = directions["all 45 (paper config)"]
for name, d in directions.items():
    c = torch.nn.functional.cosine_similarity(d, all45, dim=0).item()
    print(f"  {name:30s}  cos = {c:.3f}")


# 4. What about the non-self-ref direction specifically — is it noise or
#    something else? Compare to a "shuffled labels" null.
print("\n\nNoise control: what cosine would two random directions produce?")
hidden = pos.shape[-1]
null_cos = []
for _ in range(50):
    a = torch.randn(hidden)
    b = torch.randn(hidden)
    null_cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
null_cos = np.array(null_cos)
print(f"  Random N({hidden}) cosine: {null_cos.mean():.4f} ± {null_cos.std():.4f}")
print(f"  (If non-self-ref direction cosine with self-ref ≈ 0, that's noise.")
print(f"   If it's substantially > random baseline, it's measuring SOMETHING.)")
