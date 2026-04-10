#!/usr/bin/env python3
"""Generate per-model cross-layer analysis figures.

Produces separate figures for each model:
  - Cross-layer cosine similarity heatmap
  - Reliability-weighted similarity heatmap
  - Stride analysis (direction similarity by layer distance)

Requires activation files in /tmp/{model}_all_layers/ or adjust paths below.

Usage:
    python scripts/generate_cross_layer_figures.py
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.metrics import split_half_reliability, extract_direction, cosine_similarity

N_SELF_REF = 15
N_PROVOCATIVE = 15
N_QUESTIONS = 45
SELF_REF_SLICE = slice(0, N_SELF_REF + N_PROVOCATIVE)

MODELS = [
    ("Llama 3.3-70B-Instruct", "/tmp/llama_all_layers", "meta-llama_Llama-3.3-70B-Instruct", "llama"),
    ("Qwen 2.5-72B-Instruct", "/tmp/qwen_all_layers", "Qwen_Qwen2.5-72B-Instruct", "qwen"),
]


def load_self_ref_activations(act_dir, model_tag, layer):
    pos = torch.load(f"{act_dir}/positive_baseline_{model_tag}_layer{layer}.pt", weights_only=True)
    neg = torch.load(f"{act_dir}/negative_baseline_{model_tag}_layer{layer}.pt", weights_only=True)
    n_pairs = pos.shape[0] // N_QUESTIONS
    ps, ns = [], []
    for pi in range(n_pairs):
        start = pi * N_QUESTIONS
        ps.append(pos[start : start + N_QUESTIONS][SELF_REF_SLICE])
        ns.append(neg[start : start + N_QUESTIONS][SELF_REF_SLICE])
    return torch.cat(ps, dim=0), torch.cat(ns, dim=0)


def main():
    for model_label, act_dir, model_tag, prefix in MODELS:
        pos_files = sorted(Path(act_dir).glob(f"positive_baseline_{model_tag}_layer*.pt"))
        layers = sorted(set(int(f.stem.split("layer")[-1]) for f in pos_files))

        directions = {}
        reliabilities = {}
        for layer in layers:
            pt, nt = load_self_ref_activations(act_dir, model_tag, layer)
            directions[layer] = extract_direction(pt, nt)
            reliabilities[layer] = split_half_reliability(pt, nt, n_splits=100)

        n = len(layers)

        # Cross-layer cosine similarity matrix
        cos_matrix = np.zeros((n, n))
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                cos_matrix[i, j] = cosine_similarity(directions[l1], directions[l2])

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cos_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, xticklabels=layers, yticklabels=layers,
            ax=ax, annot_kws={"size": 5},
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"Cross-Layer Cosine Similarity — {model_label}")
        plt.tight_layout()
        plt.savefig(f"paper/{prefix}_cross_layer_cosine.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"paper/{prefix}_cross_layer_cosine.pdf", bbox_inches="tight")
        plt.close()

        # Reliability-weighted similarity
        weighted = np.zeros((n, n))
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                weighted[i, j] = cos_matrix[i, j] * reliabilities[l1] * reliabilities[l2]

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            weighted, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, xticklabels=layers, yticklabels=layers,
            ax=ax, annot_kws={"size": 5},
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"Reliability-Weighted Cross-Layer Similarity — {model_label}")
        plt.tight_layout()
        plt.savefig(f"paper/{prefix}_weighted_cross_layer.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"paper/{prefix}_weighted_cross_layer.pdf", bbox_inches="tight")
        plt.close()

        # Stride analysis
        fig, ax = plt.subplots(figsize=(8, 5))
        max_stride = 8
        for stride in range(1, max_stride + 1):
            starts = []
            cosines = []
            for i in range(n - stride):
                l1 = layers[i]
                starts.append(l1)
                cosines.append(cos_matrix[i, i + stride])
            layer_dist = layers[stride] - layers[0]
            ax.plot(starts, cosines, "o-", markersize=3, label=f"stride {stride} ({layer_dist} layers)")

        ax.set_xlabel("Starting Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Direction Similarity by Layer Distance — {model_label}")
        ax.legend(fontsize=7)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_ylim(-0.3, 1.05)
        plt.tight_layout()
        plt.savefig(f"paper/{prefix}_stride_similarity.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"paper/{prefix}_stride_similarity.pdf", bbox_inches="tight")
        plt.close()

        print(f"{model_label}: done")


if __name__ == "__main__":
    main()
