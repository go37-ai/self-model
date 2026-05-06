"""Faithful reconstruction of Figure 4a in scale-invariant units.

Data sources (matching the paper's setup):
  - Baseline Entity:  2026-04-05_0824 matched run (15 conv × 15 provocative = 225 samples)
  - Baseline Process: extraction data, sliced to same 15 conv × 15 provocative subset
  - Cap All (L4+):    2026-04-10_1635 cap_all_from_L4 (225 samples, same prompts)

Direction: extracted from full 1125-sample extraction data per layer (canonical d_L).

Three plots:
  Top:    Raw projections (paper Figure 4 style, scale-dependent)
  Middle: Z-scored projections (gap = Cohen's d, scale-invariant)
  Bottom: Cosine alignment with d_L per condition (bounded in [-1,1], scale-invariant)
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
UNCAPPED_DIR = REPO / "data/results/llama_uncapped_entity_activations"
DIRECTIONS_DIR = REPO / "data/results"  # direction_llama_layer{N}.pt — what the paper used
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
OUT_DIR = REPO / "data/results/spearman_brown"

# Matched subset: 15 conversational pairs (0-14) × 15 provocative questions (15-29)
CONV_PAIR_IDX = list(range(0, 15))
PROVOC_Q_IDX = list(range(15, 30))


def load_extract(layer: int, condition: str) -> torch.Tensor:
    """Load full extraction activations: (25 pairs × 45 questions, hidden)."""
    p = EXTRACT_DIR / f"{condition}_baseline_{MODEL}_layer{layer}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float().view(25, 45, -1)


def load_matched(d: Path, layer: int) -> torch.Tensor:
    """Load matched 225-sample run activations."""
    p = d / f"entity_layer{layer}_{MODEL}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float()


LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in EXTRACT_DIR.glob("positive_baseline_*.pt")
})


per_layer = []
for layer in LAYERS:
    # The paper figure used the stored direction files (direction_llama_layer{N}.pt),
    # not freshly extracted directions. They differ slightly (cos ≈ 0.91, ~1.45× norm).
    # We use the stored direction here to faithfully reproduce the paper figure.
    d_raw = torch.load(
        DIRECTIONS_DIR / f"direction_llama_layer{layer}.pt",
        weights_only=True, map_location="cpu",
    ).float().flatten()
    d_hat = d_raw / d_raw.norm()

    # Full-extraction pos and neg for the pooled std denominator
    pos_full = load_extract(layer, "positive").reshape(-1, 8192)  # 1125 × 8192
    neg_full = load_extract(layer, "negative").reshape(-1, 8192)

    # Per-condition pooled std for z-scoring (computed from full extraction data)
    pos_proj_full = pos_full @ d_hat
    neg_proj_full = neg_full @ d_hat
    n = len(pos_proj_full)
    pooled_std = float(np.sqrt(
        ((pos_proj_full.var(unbiased=True).item() * (n - 1))
         + (neg_proj_full.var(unbiased=True).item() * (n - 1)))
        / (2 * n - 2)
    ))

    # ---- Baseline Entity & Process (full 1125-sample extraction data) ----
    # The paper figure projects all 25 pairs × 45 questions onto the stored
    # direction at each layer for both baselines. We use the same.
    ent_acts = pos_full
    proc_acts = neg_full
    ent_proj = ent_acts @ d_hat
    proc_proj = proc_acts @ d_hat
    ent_cos = F.cosine_similarity(ent_acts, d_hat.unsqueeze(0), dim=1)
    proc_cos = F.cosine_similarity(proc_acts, d_hat.unsqueeze(0), dim=1)

    # ---- Cap All (L4+) (capping run, 225 samples) ----
    cap_acts = load_matched(CAPALL_DIR, layer)
    cap_proj = cap_acts @ d_hat
    cap_cos = F.cosine_similarity(cap_acts, d_hat.unsqueeze(0), dim=1)

    per_layer.append({
        "layer": layer,
        "pooled_std": pooled_std,
        # raw projection means (matches paper figure)
        "raw_entity": float(ent_proj.mean().item()),
        "raw_process": float(proc_proj.mean().item()),
        "raw_capall": float(cap_proj.mean().item()),
        # z-scored projections (each / pooled std)
        "z_entity": float(ent_proj.mean().item()) / pooled_std,
        "z_process": float(proc_proj.mean().item()) / pooled_std,
        "z_capall": float(cap_proj.mean().item()) / pooled_std,
        # mean cosine alignment with d_L
        "cos_entity": float(ent_cos.mean().item()),
        "cos_process": float(proc_cos.mean().item()),
        "cos_capall": float(cap_cos.mean().item()),
    })
    print(
        f"  L{layer:2d}: "
        f"entity raw={per_layer[-1]['raw_entity']:6.2f}  z={per_layer[-1]['z_entity']:+.2f}σ  "
        f"process raw={per_layer[-1]['raw_process']:6.2f}  z={per_layer[-1]['z_process']:+.2f}σ  "
        f"capall raw={per_layer[-1]['raw_capall']:6.2f}  z={per_layer[-1]['z_capall']:+.2f}σ"
    )


# Plot — three panels stacked vertically for direct visual comparison
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
layers_arr = np.array([r["layer"] for r in per_layer])

# Panel 1: raw projections (matches paper figure style)
ax = axes[0]
ax.plot(layers_arr, [r["raw_entity"] for r in per_layer], "o-",
        color="#1f77b4", label="Baseline Entity", linewidth=2)
ax.plot(layers_arr, [r["raw_process"] for r in per_layer], "s-",
        color="#2ca02c", label="Baseline Process", linewidth=2)
ax.plot(layers_arr, [r["raw_capall"] for r in per_layer], "D-",
        color="#ff9900", label="Cap All (L4+)", linewidth=2)
ax.axhline(0, color="black", lw=0.5, alpha=0.7)
ax.set_ylabel("Mean Projection (per-layer direction)")
ax.set_title("Raw projections (paper Figure 4 style — scale-dependent)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Panel 2: z-scored projections (gap = Cohen's d)
ax = axes[1]
ax.fill_between(
    layers_arr,
    [r["z_process"] for r in per_layer],
    [r["z_entity"] for r in per_layer],
    alpha=0.08, color="purple",
)
ax.plot(layers_arr, [r["z_entity"] for r in per_layer], "o-",
        color="#1f77b4", label="Baseline Entity (z)", linewidth=2)
ax.plot(layers_arr, [r["z_process"] for r in per_layer], "s-",
        color="#2ca02c", label="Baseline Process (z)", linewidth=2)
ax.plot(layers_arr, [r["z_capall"] for r in per_layer], "D-",
        color="#ff9900", label="Cap All (z)", linewidth=2)
ax.axhline(0, color="black", lw=0.5, alpha=0.7)
ax.set_ylabel("Mean projection / pooled σ at layer")
ax.set_title("Z-scored projections (scale-invariant — entity-process gap = Cohen's d)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Panel 3: Cosine alignment with d_L per condition
ax = axes[2]
ax.plot(layers_arr, [r["cos_entity"] for r in per_layer], "o-",
        color="#1f77b4", label="Baseline Entity, mean cos(act, d_L)", linewidth=2)
ax.plot(layers_arr, [r["cos_process"] for r in per_layer], "s-",
        color="#2ca02c", label="Baseline Process, mean cos(act, d_L)", linewidth=2)
ax.plot(layers_arr, [r["cos_capall"] for r in per_layer], "D-",
        color="#ff9900", label="Cap All, mean cos(act, d_L)", linewidth=2)
ax.axhline(0, color="black", lw=0.5, alpha=0.7)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean cos(activation, d_L)")
ax.set_title("Cosine alignment with d_L (scale-invariant — bounded in [-1, 1])")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)

plt.suptitle(
    "Figure 4 reconstruction: three projection metrics on the same data\n"
    "(Llama 3.3-70B, 225 samples = 15 conv × 15 provocative)",
    fontsize=13, y=0.995,
)
plt.tight_layout()

out_path = OUT_DIR / "figure4_reconstruction_three_metrics.png"
plt.savefig(out_path, dpi=140)
print(f"\nPlot saved: {out_path}")

with open(OUT_DIR / "figure4_reconstruction_per_layer.json", "w") as f:
    json.dump(per_layer, f, indent=2)
print(f"Data saved: {OUT_DIR / 'figure4_reconstruction_per_layer.json'}")
