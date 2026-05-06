"""Scale-normalized per-layer analysis for Llama 3.3-70B Cat 5.

Three layer-selection criteria, all computed on the same data:

1. Naive in-sample projection difference: ||d||² (= d · d when d = pos_mean - neg_mean)
   - Tautologically peaks at the layer with largest activation magnitude.
   - Not a meaningful signal-to-noise measure.

2. Cohen's d: (mean_pos_proj - mean_neg_proj) / pooled_std_proj
   - Standardized effect size, scale-invariant.
   - Tells you how many SDs separate the two conditions when projected onto d.

3. Cosine separation: mean(cos(pos_i, d̂)) - mean(cos(neg_i, d̂))
   - Bounded in [-1, 1], scale-invariant.
   - Tells you how much the angle between sample activations and d differs
     between conditions.

For comparison we also report split-half reliability per layer (the criterion
the paper actually uses).

Output: per-layer table + plot showing all four criteria normalized to [0, 1].
"""
from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/brian/repos/self-model")
ACT_DIR = REPO / "data/results/llama_baseline_activations"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
NUM_PAIRS = 25
NUM_QUESTIONS = 45
OUT_DIR = REPO / "data/results/spearman_brown"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(0)
np.random.seed(0)


# Discover layers
LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in ACT_DIR.glob("positive_baseline_*.pt")
})
print(f"Layers available: {LAYERS}")


def load_acts(layer: int, condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_baseline_{MODEL}_layer{layer}.pt"
    t = torch.load(p, weights_only=True, map_location="cpu").float()
    return t.view(NUM_PAIRS, NUM_QUESTIONS, -1)


def compute_layer_metrics(layer: int) -> dict:
    pos = load_acts(layer, "positive")
    neg = load_acts(layer, "negative")
    pos_flat = pos.reshape(-1, pos.shape[-1])
    neg_flat = neg.reshape(-1, neg.shape[-1])

    # Direction (raw difference of means, then unit-normalized)
    d_raw = pos_flat.mean(0) - neg_flat.mean(0)
    d_norm = d_raw / d_raw.norm()

    # 1. Raw projection difference (= ||d||²)
    raw_proj_diff = float((d_raw @ d_raw).item())

    # 2. Cohen's d on projections onto d_norm
    pos_proj = pos_flat @ d_norm  # (N_samples,)
    neg_proj = neg_flat @ d_norm
    pooled_std = float(
        np.sqrt(
            ((pos_proj.var(unbiased=True).item() * (len(pos_proj) - 1))
             + (neg_proj.var(unbiased=True).item() * (len(neg_proj) - 1)))
            / (len(pos_proj) + len(neg_proj) - 2)
        )
    )
    cohens_d = float((pos_proj.mean() - neg_proj.mean()).item()) / pooled_std

    # 3. Cosine separation: mean cosine between samples and direction, per condition
    pos_cos = F.cosine_similarity(pos_flat, d_norm.unsqueeze(0), dim=1)
    neg_cos = F.cosine_similarity(neg_flat, d_norm.unsqueeze(0), dim=1)
    cosine_separation = float((pos_cos.mean() - neg_cos.mean()).item())

    # 4. Split-half reliability (the paper's criterion)
    rs = []
    pair_indices = list(range(NUM_PAIRS))
    for _ in range(200):
        random.shuffle(pair_indices)
        a_pairs = pair_indices[:12]
        b_pairs = pair_indices[12:24]
        a_pos = pos[a_pairs].reshape(-1, pos.shape[-1])
        a_neg = neg[a_pairs].reshape(-1, neg.shape[-1])
        b_pos = pos[b_pairs].reshape(-1, pos.shape[-1])
        b_neg = neg[b_pairs].reshape(-1, neg.shape[-1])
        d_a = a_pos.mean(0) - a_neg.mean(0)
        d_b = b_pos.mean(0) - b_neg.mean(0)
        rs.append(F.cosine_similarity(d_a, d_b, dim=0).item())
    reliability = float(np.mean(rs))

    # Also report mean activation magnitude per layer (to show scale growth)
    mean_act_norm = float(((pos_flat.norm(dim=1).mean() + neg_flat.norm(dim=1).mean()) / 2).item())

    return {
        "layer": layer,
        "raw_proj_diff": raw_proj_diff,           # = ||d||²
        "d_norm_value": float(d_raw.norm().item()),  # ||d||
        "cohens_d": cohens_d,
        "cosine_separation": cosine_separation,
        "reliability": reliability,
        "mean_act_norm": mean_act_norm,
        "pos_proj_mean": float(pos_proj.mean().item()),
        "neg_proj_mean": float(neg_proj.mean().item()),
        "proj_separation_raw": float((pos_proj.mean() - neg_proj.mean()).item()),
    }


print("\nComputing per-layer metrics...")
results = []
for layer in LAYERS:
    m = compute_layer_metrics(layer)
    results.append(m)
    print(
        f"  L{layer:2d}: "
        f"||d||={m['d_norm_value']:6.2f}  "
        f"||d||²={m['raw_proj_diff']:8.2f}  "
        f"Cohen's d={m['cohens_d']:6.2f}  "
        f"cos_sep={m['cosine_separation']:.3f}  "
        f"reliability={m['reliability']:.3f}"
    )


# Save results
out_json = OUT_DIR / "llama_per_layer_criteria.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")


# Print a comparison table
print("\n" + "=" * 76)
print("Layer selection by each criterion:")
print("=" * 76)

def best_layer(key, results, transform=lambda x: x):
    best = max(results, key=lambda r: transform(r[key]))
    return best["layer"], transform(best[key])

raw_layer, raw_val = best_layer("raw_proj_diff", results)
cohen_layer, cohen_val = best_layer("cohens_d", results)
cos_layer, cos_val = best_layer("cosine_separation", results)
rel_layer, rel_val = best_layer("reliability", results)

print(f"  Naive ||d||²:           L{raw_layer}  (||d||²={raw_val:.2f})")
print(f"  Cohen's d:              L{cohen_layer}  (d={cohen_val:.2f})")
print(f"  Cosine separation:      L{cos_layer}  (cos_sep={cos_val:.3f})")
print(f"  Split-half reliability: L{rel_layer}  (r={rel_val:.3f})  ← paper's choice")
print(f"  Lu et al. target layer: L40  (pre-specified, no criterion documented)")


# Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
layers_arr = np.array([r["layer"] for r in results])

# Subplot 1: Raw ||d||² (the bad one)
ax = axes[0, 0]
ax.plot(layers_arr, [r["raw_proj_diff"] for r in results], "o-", color="C3")
ax.axvline(raw_layer, color="C3", linestyle="--", alpha=0.5,
            label=f"Peak: L{raw_layer}")
ax.set_xlabel("Layer")
ax.set_ylabel("||d||² (raw projection difference)")
ax.set_title("Naive in-sample projection difference\n(dominated by residual stream scale)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Subplot 2: Cohen's d
ax = axes[0, 1]
ax.plot(layers_arr, [r["cohens_d"] for r in results], "o-", color="C2")
ax.axvline(cohen_layer, color="C2", linestyle="--", alpha=0.5,
            label=f"Peak: L{cohen_layer}")
ax.axvline(20, color="C0", linestyle=":", alpha=0.5, label="Paper L20")
ax.axvline(40, color="C4", linestyle=":", alpha=0.5, label="Lu et al. L40")
ax.set_xlabel("Layer")
ax.set_ylabel("Cohen's d (standardized projection separation)")
ax.set_title("Scale-normalized: Cohen's d on projections")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Subplot 3: Cosine separation
ax = axes[1, 0]
ax.plot(layers_arr, [r["cosine_separation"] for r in results], "o-", color="C1")
ax.axvline(cos_layer, color="C1", linestyle="--", alpha=0.5,
            label=f"Peak: L{cos_layer}")
ax.axvline(20, color="C0", linestyle=":", alpha=0.5, label="Paper L20")
ax.axvline(40, color="C4", linestyle=":", alpha=0.5, label="Lu et al. L40")
ax.set_xlabel("Layer")
ax.set_ylabel("Cosine separation")
ax.set_title("Scale-normalized: cosine separation between conditions")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Subplot 4: Split-half reliability
ax = axes[1, 1]
ax.plot(layers_arr, [r["reliability"] for r in results], "o-", color="C0")
ax.axvline(rel_layer, color="C0", linestyle="--", alpha=0.5,
            label=f"Peak: L{rel_layer}")
ax.axvline(40, color="C4", linestyle=":", alpha=0.5, label="Lu et al. L40")
ax.set_xlabel("Layer")
ax.set_ylabel("Split-half reliability (cosine)")
ax.set_title("Paper's criterion: split-half reliability")
ax.legend(loc="lower right")
ax.set_ylim(0.5, 1.0)
ax.grid(alpha=0.3)

plt.suptitle(
    f"Llama 3.3-70B Cat 5 baseline: layer selection criteria\n"
    f"(25 pairs × 45 questions, computed on extraction data)",
    fontsize=13,
)
plt.tight_layout()
plot_path = OUT_DIR / "llama_layer_selection_criteria.png"
plt.savefig(plot_path, dpi=140)
print(f"\nPlot saved: {plot_path}")
