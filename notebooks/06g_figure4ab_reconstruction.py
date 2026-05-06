"""Reconstruct paper Figure 4a (CapAll) and Figure 4b (Cap@L72) in three units.

Layout: 2 figure rows × 3 metric columns

         Raw projections     Z-scored (Cohen's d)     Cosine alignment
Fig 4a:  baseline + CapAll   baseline + CapAll        baseline + CapAll
Fig 4b:  baseline + Cap@L72  baseline + Cap@L72       baseline + Cap@L72

Data sources:
  - Baselines (entity, process): full 1125-sample extraction data
  - CapAll (L4+):  225 samples from 2026-04-10_1635/cap_all_from_L4/
  - Cap@L72:       225 samples from 2026-04-05_0942/cap_L72/
  - Direction: stored direction_llama_layer{N}.pt (matches paper)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/brian/repos/self-model")
EXTRACT_DIR = REPO / "data/results/llama_baseline_activations"
CAPALL_DIR = REPO / "data/results/llama_capall_activations"
CAPL72_DIR = REPO / "data/results/llama_capL72_activations"
DIRECTIONS_DIR = REPO / "data/results"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
OUT_DIR = REPO / "data/results/spearman_brown"


def load_extract(layer, condition):
    p = EXTRACT_DIR / f"{condition}_baseline_{MODEL}_layer{layer}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float()


def load_cap(d, layer):
    p = d / f"entity_layer{layer}_{MODEL}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float()


LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in EXTRACT_DIR.glob("positive_baseline_*.pt")
})


per_layer = []
for layer in LAYERS:
    d_raw = torch.load(
        DIRECTIONS_DIR / f"direction_llama_layer{layer}.pt",
        weights_only=True, map_location="cpu",
    ).float().flatten()
    d_hat = d_raw / d_raw.norm()

    pos_full = load_extract(layer, "positive")  # (1125, 8192)
    neg_full = load_extract(layer, "negative")
    capall_acts = load_cap(CAPALL_DIR, layer)
    capL72_acts = load_cap(CAPL72_DIR, layer)

    # Pooled std for z-scoring (from full extraction data)
    pos_proj = pos_full @ d_hat
    neg_proj = neg_full @ d_hat
    n = len(pos_proj)
    pooled = float(np.sqrt(
        ((pos_proj.var(unbiased=True).item() * (n - 1))
         + (neg_proj.var(unbiased=True).item() * (n - 1)))
        / (2 * n - 2)
    ))

    capall_proj = capall_acts @ d_hat
    capL72_proj = capL72_acts @ d_hat

    per_layer.append({
        "layer": layer,
        "pooled_std": pooled,
        "raw_entity": float(pos_proj.mean().item()),
        "raw_process": float(neg_proj.mean().item()),
        "raw_capall": float(capall_proj.mean().item()),
        "raw_capL72": float(capL72_proj.mean().item()),
        "z_entity": float(pos_proj.mean().item()) / pooled,
        "z_process": float(neg_proj.mean().item()) / pooled,
        "z_capall": float(capall_proj.mean().item()) / pooled,
        "z_capL72": float(capL72_proj.mean().item()) / pooled,
        "cos_entity": float(F.cosine_similarity(pos_full, d_hat.unsqueeze(0), dim=1).mean().item()),
        "cos_process": float(F.cosine_similarity(neg_full, d_hat.unsqueeze(0), dim=1).mean().item()),
        "cos_capall": float(F.cosine_similarity(capall_acts, d_hat.unsqueeze(0), dim=1).mean().item()),
        "cos_capL72": float(F.cosine_similarity(capL72_acts, d_hat.unsqueeze(0), dim=1).mean().item()),
    })
    print(
        f"  L{layer:2d}: ent={per_layer[-1]['raw_entity']:6.2f}  "
        f"proc={per_layer[-1]['raw_process']:6.2f}  "
        f"capall={per_layer[-1]['raw_capall']:6.2f}  "
        f"capL72={per_layer[-1]['raw_capL72']:6.2f}"
    )


fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
layers_arr = np.array([r["layer"] for r in per_layer])

ENT_C, PROC_C, CAPALL_C, CAPL72_C = "#1f77b4", "#2ca02c", "#ff9900", "#8B4513"


def plot_baselines_and_intervention(ax, intervention_key, intervention_label,
                                     intervention_color, ylabel, title,
                                     metric_prefix, shade=True, intervention_style="-"):
    if shade:
        ax.fill_between(
            layers_arr,
            [r[f"{metric_prefix}_process"] for r in per_layer],
            [r[f"{metric_prefix}_entity"] for r in per_layer],
            alpha=0.08, color="purple",
        )
    ax.plot(layers_arr, [r[f"{metric_prefix}_entity"] for r in per_layer], "o-",
            color=ENT_C, label="Baseline Entity", linewidth=2)
    ax.plot(layers_arr, [r[f"{metric_prefix}_process"] for r in per_layer], "s-",
            color=PROC_C, label="Baseline Process", linewidth=2)
    ax.plot(layers_arr, [r[intervention_key] for r in per_layer],
            "D" + intervention_style, color=intervention_color,
            label=intervention_label, linewidth=2,
            linestyle=("--" if intervention_label.startswith("Cap@") else "-"))
    ax.axhline(0, color="black", lw=0.5, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)


# Row 1: Figure 4a (CapAll)
plot_baselines_and_intervention(
    axes[0, 0], "raw_capall", "Cap All (L4+)", CAPALL_C,
    "Mean Projection (raw)", "Figure 4a — Raw projections (paper style)",
    "raw", shade=False,
)
plot_baselines_and_intervention(
    axes[0, 1], "z_capall", "Cap All (L4+)", CAPALL_C,
    "z = mean projection / pooled σ", "Figure 4a — Z-scored (gap = Cohen's d)",
    "z", shade=True,
)
plot_baselines_and_intervention(
    axes[0, 2], "cos_capall", "Cap All (L4+)", CAPALL_C,
    "Mean cos(activation, d_L)", "Figure 4a — Cosine alignment",
    "cos", shade=False,
)

# Row 2: Figure 4b (Cap@L72)
plot_baselines_and_intervention(
    axes[1, 0], "raw_capL72", "Cap@L72", CAPL72_C,
    "Mean Projection (raw)", "Figure 4b — Raw projections (paper style)",
    "raw", shade=False,
)
plot_baselines_and_intervention(
    axes[1, 1], "z_capL72", "Cap@L72", CAPL72_C,
    "z = mean projection / pooled σ", "Figure 4b — Z-scored (gap = Cohen's d)",
    "z", shade=True,
)
plot_baselines_and_intervention(
    axes[1, 2], "cos_capL72", "Cap@L72", CAPL72_C,
    "Mean cos(activation, d_L)", "Figure 4b — Cosine alignment",
    "cos", shade=False,
)

axes[1, 0].set_xlabel("Layer")
axes[1, 1].set_xlabel("Layer")
axes[1, 2].set_xlabel("Layer")

plt.suptitle(
    "Figures 4a (CapAll) and 4b (Cap@L72): three projection metrics on the same data\n"
    "(Llama 3.3-70B; baselines = 1125 extraction samples, interventions = 225 matched samples)",
    fontsize=13, y=0.995,
)
plt.tight_layout()

out_path = OUT_DIR / "figure4ab_reconstruction.png"
plt.savefig(out_path, dpi=140)
print(f"\nPlot saved: {out_path}")

with open(OUT_DIR / "figure4ab_per_layer.json", "w") as f:
    json.dump(per_layer, f, indent=2)
print(f"Data saved: {OUT_DIR / 'figure4ab_per_layer.json'}")
