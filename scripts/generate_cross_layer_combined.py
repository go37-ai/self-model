"""Generate per-model cross-layer combined figure (single shared colorbar).

For each model, produces a side-by-side figure:
  Left:  Cross-layer cosine similarity heatmap
  Right: Reliability-weighted cross-layer similarity heatmap
Both halves share one colorbar (vmin=-1, vmax=1, cmap RdBu_r).

Self-ref-only subset (30 questions per pair: 15 neutral + 15 provocative),
matching the abstract's headline reliability convention.

Usage:
    python scripts/generate_cross_layer_combined.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.metrics import cosine_similarity, extract_direction, split_half_reliability

N_QUESTIONS = 45
N_SELF_REF = 30

CONFIG = [
    ("Llama 3.3-70B-Instruct", "meta-llama_Llama-3.3-70B-Instruct",
     ROOT / "data" / "results" / "llama_baseline_activations",
     "{cond}_baseline_{name}_layer{L}.pt", "llama"),
    ("Qwen 2.5-72B-Instruct", "Qwen_Qwen2.5-72B-Instruct",
     ROOT / "data" / "results" / "1.1_naive_72b_v2" / "activations",
     "{cond}_naive_{name}_layer{L}.pt", "qwen"),
]


def load_self_ref(act_dir: Path, pattern: str, name: str, layer: int):
    pos = torch.load(act_dir / pattern.format(cond="positive", name=name, L=layer), weights_only=True)
    neg = torch.load(act_dir / pattern.format(cond="negative", name=name, L=layer), weights_only=True)
    n_pairs = pos.shape[0] // N_QUESTIONS
    ps, ns = [], []
    for pi in range(n_pairs):
        start = pi * N_QUESTIONS
        ps.append(pos[start:start + N_QUESTIONS][:N_SELF_REF])
        ns.append(neg[start:start + N_QUESTIONS][:N_SELF_REF])
    return torch.cat(ps), torch.cat(ns)


def discover_layers(act_dir: Path, pattern: str, name: str):
    glob = pattern.format(cond="positive", name=name, L="*")
    return sorted(int(p.stem.split("layer")[-1]) for p in act_dir.glob(glob))


def annotate(ax, M, fontsize=4):
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=fontsize)


def main():
    for label, name, act_dir, pattern, prefix in CONFIG:
        layers = discover_layers(act_dir, pattern, name)
        if not layers:
            print(f"[skip] no activations in {act_dir}")
            continue
        print(f"{label}: {len(layers)} layers")

        directions, reliab = {}, {}
        for L in layers:
            p, n = load_self_ref(act_dir, pattern, name, L)
            directions[L] = extract_direction(p, n)
            reliab[L] = split_half_reliability(p, n, n_splits=100)

        N = len(layers)
        cos_M = np.zeros((N, N))
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                cos_M[i, j] = cosine_similarity(directions[l1], directions[l2])
        weighted = cos_M * np.outer([reliab[L] for L in layers], [reliab[L] for L in layers])

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        for ax, M, title in [(axes[0], cos_M, "Cross-layer cosine similarity"),
                             (axes[1], weighted, "Reliability-weighted similarity")]:
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(N))
            ax.set_xticklabels(layers, rotation=45, fontsize=7)
            ax.set_yticks(range(N))
            ax.set_yticklabels(layers, fontsize=7)
            annotate(ax, M)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")
            ax.set_title(title)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
        cbar.set_label("similarity")
        fig.suptitle(label)

        out = ROOT / "paper" / f"{prefix}_cross_layer_combined"
        fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        print(f"  wrote {out.with_suffix('.png')} + .pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
