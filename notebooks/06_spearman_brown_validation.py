"""Empirical validation of Spearman-Brown for contrastive direction reliability.

Question: does the Spearman-Brown prophecy formula correctly predict how
split-half reliability of an extracted direction scales with the number of
contrastive pairs?

Approach:
1. Load Qwen 2.5-72B Cat 5 activations (25 pairs × 45 questions × hidden).
2. For N in {5, 10, 15, 20, 25} sub-sample many random pair subsets,
   compute split-half cosine reliability of the extracted direction,
   and average across subsamples.
3. Compare the empirical r(N) curve to the Spearman-Brown prophecy from r(5).
4. If validated, apply formula to Cat 1-4 (small-N) reliability to predict
   what their reliability would be at N=25.

Usage:  python notebooks/06_spearman_brown_validation.py
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/home/brian/repos/self-model")
ACT_DIR = REPO / "data/results/llama_baseline_activations"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
LAYER = 20
NUM_PAIRS = 25
NUM_QUESTIONS = 45
OUT_DIR = REPO / "data/results/spearman_brown"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# Step 1: load activations and reshape per-pair
# ---------------------------------------------------------------------------
def load_acts(condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_baseline_{MODEL}_layer{LAYER}.pt"
    t = torch.load(p, weights_only=True, map_location="cpu").float()
    # (N_pairs * N_questions, hidden) -> (N_pairs, N_questions, hidden)
    return t.view(NUM_PAIRS, NUM_QUESTIONS, -1)


pos = load_acts("positive")
neg = load_acts("negative")
print(f"Loaded activations: pos {tuple(pos.shape)}, neg {tuple(neg.shape)}")


def direction_from_pairs(pair_idx: list[int]) -> torch.Tensor:
    """Extract a self-reification direction from a subset of pair indices.

    Mirrors the production extraction: mean activation difference between
    positive and negative conditions over all (pair, question) samples.
    """
    p_sub = pos[pair_idx].reshape(-1, pos.shape[-1])
    n_sub = neg[pair_idx].reshape(-1, neg.shape[-1])
    return p_sub.mean(0) - n_sub.mean(0)


def split_half_reliability(pair_idx: list[int]) -> float:
    """Compute split-half reliability for a list of pair indices.

    Splits the pairs into two halves, extracts a direction from each,
    returns cosine similarity.
    """
    pair_idx = list(pair_idx)
    random.shuffle(pair_idx)
    half = len(pair_idx) // 2
    a = direction_from_pairs(pair_idx[:half])
    b = direction_from_pairs(pair_idx[half : 2 * half])
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# ---------------------------------------------------------------------------
# Step 2: empirical reliability curve via subsampling
# ---------------------------------------------------------------------------
N_VALUES = [4, 6, 8, 10, 12, 16, 20, 24]
N_TRIALS_PER_N = 200

empirical = {}
all_indices = list(range(NUM_PAIRS))

print("\nEmpirical reliability curve (split-half cosine, mean ± std):")
for n in N_VALUES:
    rs = []
    for _ in range(N_TRIALS_PER_N):
        sample = random.sample(all_indices, n)
        rs.append(split_half_reliability(sample))
    rs = np.array(rs)
    empirical[n] = (rs.mean(), rs.std(), rs)
    print(f"  N={n:2d}  r = {rs.mean():.4f} ± {rs.std():.4f}")


# ---------------------------------------------------------------------------
# Step 3: Spearman-Brown extrapolation from smallest N
# ---------------------------------------------------------------------------
def spearman_brown(r_base: float, k: float) -> float:
    """Predict r at k× test length given r at base length."""
    return (k * r_base) / (1 + (k - 1) * r_base)


def invert_spearman_brown(r_observed: float, k: float) -> float:
    """Given r at k× test length, recover r at base length.

    r_base = r_observed / (k - (k - 1) * r_observed)
    """
    return r_observed / (k - (k - 1) * r_observed)


N_BASE = N_VALUES[0]
r_base_mean, _, _ = empirical[N_BASE]

print(f"\nSpearman-Brown prediction from N={N_BASE} (r={r_base_mean:.4f}):")
predicted = {N_BASE: r_base_mean}
for n in N_VALUES[1:]:
    k = n / N_BASE
    predicted[n] = spearman_brown(r_base_mean, k)
    emp_mean = empirical[n][0]
    delta = predicted[n] - emp_mean
    print(
        f"  N={n:2d}  predicted r = {predicted[n]:.4f}  "
        f"empirical = {emp_mean:.4f}  Δ = {delta:+.4f}"
    )


# ---------------------------------------------------------------------------
# Step 4: fit and report quality
# ---------------------------------------------------------------------------
emp_means = np.array([empirical[n][0] for n in N_VALUES])
pred = np.array([predicted[n] for n in N_VALUES])
rmse = float(np.sqrt(np.mean((emp_means - pred) ** 2)))
print(f"\nRMSE (Spearman-Brown vs empirical) = {rmse:.4f}")
print("Spearman-Brown is considered well-fit if RMSE < 0.03.")


# ---------------------------------------------------------------------------
# Step 5: apply validated formula to small-N category data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Cat 1-4 N=25 predictions (using validated Spearman-Brown, k=5):")
print("=" * 70)

categories_observed = {
    "Qwen 2.5-7B  Cat 1 (narrative)": (0.27, 5),
    "Qwen 2.5-7B  Cat 2 (bounded)": (0.13, 5),
    "Qwen 2.5-7B  Cat 3 (stakes)": (0.29, 5),
    "Qwen 2.5-7B  Cat 4 (observer)": (0.62, 5),
    "Qwen3-32B    Cat 1 (narrative)": (0.071, 5),
    "Qwen3-32B    Cat 2 (bounded)": (0.021, 5),
    "Qwen3-32B    Cat 3 (stakes)": (0.092, 5),
    "Qwen3-32B    Cat 4 (observer)": (0.148, 5),
}

print(f"\n  {'Category':40s}  r at N=5  predicted r at N=25")
print(f"  {'-' * 40}  --------  --------------------")
for label, (r5, n5) in categories_observed.items():
    r25 = spearman_brown(r5, 25 / n5)
    print(f"  {label:40s}  {r5:6.3f}    {r25:6.3f}")


# ---------------------------------------------------------------------------
# Step 6: plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Empirical with error bars
emp_stds = np.array([empirical[n][1] for n in N_VALUES])
ax.errorbar(
    N_VALUES, emp_means, yerr=emp_stds,
    fmt="o", capsize=4, label=f"Empirical (Cat 5, Llama 3.3-70B L{LAYER})", color="C0",
)

# Spearman-Brown curve from N=N_BASE
n_grid = np.linspace(2, 30, 100)
sb_curve = [spearman_brown(r_base_mean, x / N_BASE) for x in n_grid]
ax.plot(
    n_grid, sb_curve, "--", color="C1",
    label=f"Spearman-Brown extrapolation from N={N_BASE} (r={r_base_mean:.3f})",
)

# Cat 4 prediction marker (Qwen 7B)
r5_cat4_7b, _ = categories_observed["Qwen 2.5-7B  Cat 4 (observer)"]
r25_cat4_7b = spearman_brown(r5_cat4_7b, 5)
ax.plot([5], [r5_cat4_7b], "s", color="C2", markersize=9, label="Qwen 7B Cat 4 observed (N=5)")
ax.plot([25], [r25_cat4_7b], "*", color="C2", markersize=15,
        label=f"Qwen 7B Cat 4 predicted at N=25 (r={r25_cat4_7b:.2f})")

# Annotate Cat 5 paper anchors
ax.axhline(0.93, color="gray", linestyle=":", alpha=0.6)
ax.text(2.2, 0.94, "Paper Llama 70B Cat 5: r=0.93", fontsize=9, color="gray")
ax.axhline(0.71, color="gray", linestyle=":", alpha=0.6)
ax.text(2.2, 0.72, "Paper Qwen 72B Cat 5: r=0.71", fontsize=9, color="gray")

ax.set_xlabel("Number of contrastive pairs (N)")
ax.set_ylabel("Split-half reliability (cosine)")
ax.set_title(f"Spearman-Brown empirical validation\n(Llama 3.3-70B Cat 5 baseline activations, layer {LAYER})")
ax.set_ylim(0, 1.0)
ax.set_xlim(0, 30)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plot_path = OUT_DIR / "spearman_brown_validation.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=140)
print(f"\nPlot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Step 7: persist results
# ---------------------------------------------------------------------------
results = {
    "model": MODEL,
    "layer": LAYER,
    "num_pairs_total": NUM_PAIRS,
    "num_questions": NUM_QUESTIONS,
    "n_trials_per_n": N_TRIALS_PER_N,
    "empirical": {
        str(n): {"mean": float(empirical[n][0]), "std": float(empirical[n][1])}
        for n in N_VALUES
    },
    "spearman_brown_from_n_base": {
        "n_base": N_BASE,
        "r_base": float(r_base_mean),
        "predicted": {str(n): float(predicted[n]) for n in N_VALUES},
    },
    "rmse_sb_vs_empirical": rmse,
    "category_n25_predictions": {
        label: {
            "r_observed_n5": r5,
            "r_predicted_n25": spearman_brown(r5, 25 / n5),
        }
        for label, (r5, n5) in categories_observed.items()
    },
}

results_path = OUT_DIR / "spearman_brown_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {results_path}")
