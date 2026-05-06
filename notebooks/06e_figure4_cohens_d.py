"""Recreate Figure 4-style per-layer projection plot in scale-invariant units.

Two scale-invariant alternatives plotted side by side:
  Left:   Cohen's d at each layer (entity vs process), computed from extraction data
  Right:  Mean cosine alignment with d_L at each layer, per condition

Both versions use the 25 pairs × 45 questions extraction activations (1125
samples per condition) we already have on disk for Llama 3.3-70B.

This is the entity-vs-process baseline. CapAll line would require downloading
the cap_all_from_L4 activations from S3 (~77 MB) — adding it is left for a
follow-up if the user wants the full Figure 4 reconstruction.
"""
from __future__ import annotations

import json
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


def load_acts(layer: int, condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_baseline_{MODEL}_layer{layer}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float()


LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in ACT_DIR.glob("positive_baseline_*.pt")
})
print(f"Layers: {LAYERS}")


per_layer = []
for layer in LAYERS:
    pos_flat = load_acts(layer, "positive")  # (1125, hidden)
    neg_flat = load_acts(layer, "negative")

    # Direction at this layer (unit-normalized)
    d_raw = pos_flat.mean(0) - neg_flat.mean(0)
    d_hat = d_raw / d_raw.norm()

    # Projections onto unit direction
    pos_proj = pos_flat @ d_hat
    neg_proj = neg_flat @ d_hat

    # Cohen's d
    n_p, n_n = len(pos_proj), len(neg_proj)
    pooled = float(np.sqrt(
        ((pos_proj.var(unbiased=True).item() * (n_p - 1))
         + (neg_proj.var(unbiased=True).item() * (n_n - 1)))
        / (n_p + n_n - 2)
    ))
    cohens_d = float((pos_proj.mean() - neg_proj.mean()).item()) / pooled

    # Z-scored mean projections (each condition expressed in pooled std units)
    z_pos_mean = float(pos_proj.mean().item()) / pooled
    z_neg_mean = float(neg_proj.mean().item()) / pooled

    # Mean cosine alignment with d_L per condition (independent normalization)
    pos_cos = F.cosine_similarity(pos_flat, d_hat.unsqueeze(0), dim=1)
    neg_cos = F.cosine_similarity(neg_flat, d_hat.unsqueeze(0), dim=1)
    pos_cos_mean = float(pos_cos.mean().item())
    neg_cos_mean = float(neg_cos.mean().item())

    per_layer.append({
        "layer": layer,
        "cohens_d": cohens_d,
        "z_pos_mean": z_pos_mean,
        "z_neg_mean": z_neg_mean,
        "pos_cos_mean": pos_cos_mean,
        "neg_cos_mean": neg_cos_mean,
        "raw_pos_mean": float(pos_proj.mean().item()),
        "raw_neg_mean": float(neg_proj.mean().item()),
        "pooled_std": pooled,
    })
    print(
        f"  L{layer:2d}: cohens_d={cohens_d:5.2f}  "
        f"z_entity={z_pos_mean:+5.2f}σ  z_process={z_neg_mean:+5.2f}σ  "
        f"cos_entity={pos_cos_mean:+.3f}  cos_process={neg_cos_mean:+.3f}"
    )


# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
layers_arr = np.array([r["layer"] for r in per_layer])

# Panel 1: raw projection (the current Figure 4 style)
ax = axes[0]
ax.plot(layers_arr, [r["raw_pos_mean"] for r in per_layer], "o-",
        color="#1f77b4", label="Entity prompt (raw)")
ax.plot(layers_arr, [r["raw_neg_mean"] for r in per_layer], "o-",
        color="#2ca02c", label="Process prompt (raw)")
ax.set_xlabel("Layer")
ax.set_ylabel("Mean projection (raw units)")
ax.set_title("Current Figure 4 style: raw projections\n(scale grows with layer depth)")
ax.legend(loc="upper left")
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3)

# Panel 2: Z-scored projections (each condition mean / pooled_std at that layer)
ax = axes[1]
ax.plot(layers_arr, [r["z_pos_mean"] for r in per_layer], "o-",
        color="#1f77b4", label="Entity (z-units)")
ax.plot(layers_arr, [r["z_neg_mean"] for r in per_layer], "o-",
        color="#2ca02c", label="Process (z-units)")
# Cohen's d = the gap between the two lines
ax.fill_between(
    layers_arr,
    [r["z_neg_mean"] for r in per_layer],
    [r["z_pos_mean"] for r in per_layer],
    alpha=0.15, color="purple", label="Cohen's d gap",
)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean projection / pooled σ at layer (z-units)")
ax.set_title("Scale-invariant: z-scored projections\n(gap between lines = Cohen's d)")
ax.legend(loc="upper right")
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3)

# Panel 3: Cosine alignment with d_L per condition
ax = axes[2]
ax.plot(layers_arr, [r["pos_cos_mean"] for r in per_layer], "o-",
        color="#1f77b4", label="Entity, mean cos(act, d_L)")
ax.plot(layers_arr, [r["neg_cos_mean"] for r in per_layer], "o-",
        color="#2ca02c", label="Process, mean cos(act, d_L)")
ax.set_xlabel("Layer")
ax.set_ylabel("Mean cos(activation, d_L)")
ax.set_title("Scale-invariant: cosine alignment with d_L\n(bounded in [-1, 1])")
ax.legend(loc="lower right")
ax.axhline(0, color="black", lw=0.5)
ax.grid(alpha=0.3)

plt.suptitle(
    "Figure 4 alternatives: scale-invariant per-layer entity/process separation\n"
    "(Llama 3.3-70B Cat 5, 25 pairs × 45 questions extraction data)",
    fontsize=12,
)
plt.tight_layout()

out_path = OUT_DIR / "figure4_scale_invariant_alternatives.png"
plt.savefig(out_path, dpi=140)
print(f"\nPlot saved: {out_path}")

# Save data
with open(OUT_DIR / "figure4_per_layer.json", "w") as f:
    json.dump(per_layer, f, indent=2)
print(f"Data saved: {OUT_DIR / 'figure4_per_layer.json'}")
