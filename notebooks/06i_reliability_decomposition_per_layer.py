"""Per-layer split-half reliability decomposition for Llama 3.3-70B.

Computes reliability for each (register, question_type) combination at every
recorded layer, then plots Option 1: one panel with 4 question-type lines
(combined register) and a shaded band per line spanning min/max of
conversational and philosophical registers.

Uses the same `split_half_reliability` function from src/utils/metrics.py
as the paper's headline numbers, so this analysis is consistent with the
existing 4.1.3 reliability decomposition table at L20.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import argparse

REPO = Path("/home/brian/repos/self-model")
sys.path.insert(0, str(REPO))
from src.utils.metrics import split_half_reliability  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["llama", "qwen"], default="llama")
args = parser.parse_args()

if args.model == "llama":
    ACT_DIR = REPO / "data/results/llama_baseline_activations"
    MODEL = "meta-llama_Llama-3.3-70B-Instruct"
    ACT_PREFIX = "baseline"
    LABEL = "Llama 3.3-70B"
    OUT_PREFIX = "llama"
else:  # qwen
    ACT_DIR = REPO / "data/results/1.1_naive_72b_v2/activations"
    MODEL = "Qwen_Qwen2.5-72B-Instruct"
    ACT_PREFIX = "naive"
    LABEL = "Qwen 2.5-72B"
    OUT_PREFIX = "qwen"

NUM_PAIRS = 25
NUM_QUESTIONS = 45
OUT_DIR = REPO / "data/results/spearman_brown"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pair indexing
CONV_PAIRS = list(range(0, 15))
PHIL_PAIRS = list(range(15, 25))
ALL_PAIRS = list(range(25))

# Question subsets
Q_NEUTRAL = list(range(0, 15))
Q_PROVOC = list(range(15, 30))
Q_NONSR = list(range(30, 45))
Q_ALL_SR = Q_NEUTRAL + Q_PROVOC

REGISTERS = {
    "Conversational": CONV_PAIRS,
    "Philosophical": PHIL_PAIRS,
    "Combined": ALL_PAIRS,
}

QUESTION_TYPES = {
    "Provocative": Q_PROVOC,
    "Neutral": Q_NEUTRAL,
    "All self-ref": Q_ALL_SR,
    "Non-self-ref": Q_NONSR,
}


def load_acts(layer: int, condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_{ACT_PREFIX}_{MODEL}_layer{layer}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float().view(NUM_PAIRS, NUM_QUESTIONS, -1)


def slice_acts(acts: torch.Tensor, pair_idx: list[int], q_idx: list[int]) -> torch.Tensor:
    """Slice activations to (pairs × questions) and flatten to (n_samples, hidden)."""
    return acts[pair_idx][:, q_idx, :].reshape(-1, acts.shape[-1])


LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in ACT_DIR.glob(f"positive_{ACT_PREFIX}_*.pt")
})

# Compute reliability for every (layer, register, question_type) cell
results = {}
for layer in LAYERS:
    pos = load_acts(layer, "positive")
    neg = load_acts(layer, "negative")
    results[layer] = {}
    for reg_name, pair_idx in REGISTERS.items():
        results[layer][reg_name] = {}
        for qt_name, q_idx in QUESTION_TYPES.items():
            p_slice = slice_acts(pos, pair_idx, q_idx)
            n_slice = slice_acts(neg, pair_idx, q_idx)
            r = split_half_reliability(p_slice, n_slice, n_splits=100, seed=42)
            results[layer][reg_name][qt_name] = r
    print(
        f"L{layer:2d}: "
        + " | ".join(
            f"{qt[:5]} (cmb={results[layer]['Combined'][qt]:.2f})"
            for qt in QUESTION_TYPES
        )
    )

# Save data
with open(OUT_DIR / f"{OUT_PREFIX}_reliability_decomposition.json", "w") as f:
    json.dump({str(L): results[L] for L in LAYERS}, f, indent=2)


layers_arr = np.array(LAYERS)

# ---------- Graph 1: question-type decomposition (combined register only) ----------
QT_COLORS = {
    "Provocative": "#d62728",
    "Neutral": "#1f77b4",
    "All self-ref": "#9467bd",
    "Non-self-ref": "#7f7f7f",
}

fig, ax = plt.subplots(figsize=(10, 6))
for qt_name, color in QT_COLORS.items():
    combined = np.array([results[L]["Combined"][qt_name] for L in LAYERS])
    ax.plot(layers_arr, combined, "o-", color=color, label=qt_name, linewidth=2)

ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5)
ax.text(80, 0.71, "r = 0.7", fontsize=8, color="gray", ha="right")
ax.set_xlabel("Layer")
ax.set_ylabel("Split-half reliability")
ax.set_title(f"{LABEL}: split-half reliability by question type (combined register)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(-0.15, 1.0)
ax.grid(alpha=0.3)
plt.tight_layout()
plot_path_1 = OUT_DIR / f"{OUT_PREFIX}_reliability_by_question_type.png"
plt.savefig(plot_path_1, dpi=140)
plt.close(fig)
print(f"\nGraph 1 saved: {plot_path_1}")

# ---------- Graph 2: register decomposition (all self-ref only) ----------
REG_COLORS = {
    "Conversational": "#2ca02c",
    "Philosophical": "#ff7f0e",
    "Combined": "#9467bd",
}

fig, ax = plt.subplots(figsize=(10, 6))
for reg_name, color in REG_COLORS.items():
    series = np.array([results[L][reg_name]["All self-ref"] for L in LAYERS])
    style = "o-" if reg_name == "Combined" else "s--"
    lw = 2.5 if reg_name == "Combined" else 1.8
    ax.plot(layers_arr, series, style, color=color, label=reg_name, linewidth=lw)

ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5)
ax.text(80, 0.71, "r = 0.7", fontsize=8, color="gray", ha="right")
ax.set_xlabel("Layer")
ax.set_ylabel("Split-half reliability")
ax.set_title(f"{LABEL}: split-half reliability by register (all self-ref questions)")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(-0.15, 1.0)
ax.grid(alpha=0.3)
plt.tight_layout()
plot_path_2 = OUT_DIR / f"{OUT_PREFIX}_reliability_by_register.png"
plt.savefig(plot_path_2, dpi=140)
plt.close(fig)
print(f"Graph 2 saved: {plot_path_2}")
