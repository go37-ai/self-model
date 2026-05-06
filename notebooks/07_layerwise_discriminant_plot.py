#!/usr/bin/env python3
"""Plot per-layer formality and confidence cosines (replaces Table 2 / 4.2.4).

Reads:
  data/results/layerwise_discriminant/layerwise_discriminant_{model}.json

Writes:
  paper/figure_layerwise_discriminant_llama.{png,pdf}
  paper/figure_layerwise_discriminant_qwen.{png,pdf}

Each figure has two lines per panel (formality, confidence) with a horizontal
reference at +/-0.8 (the discriminant-validity threshold) and a marker on the
best-reliability layer.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

# Map UI label -> (file model_name, best layer in main analysis, total layers)
MODELS = {
    "Llama 3.3-70B": ("meta-llama_Llama-3.3-70B-Instruct", 20, 80),
    "Qwen 2.5-72B":  ("Qwen_Qwen2.5-72B-Instruct",         60, 80),
}


def plot_model(label: str, file_stem: str, best_layer: int, num_layers: int,
               out_path: Path) -> None:
    json_path = RESULTS / f"layerwise_discriminant_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    layers = sorted(int(L) for L in data["per_layer"].keys())
    formality = [data["per_layer"][str(L)]["formality_cosine"] for L in layers]
    confidence = [data["per_layer"][str(L)]["confidence_cosine"] for L in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0,  color="black", lw=0.5, alpha=0.4)
    ax.axhline(+0.8, color="red", lw=0.8, ls="--", alpha=0.5, label="discriminant threshold (±0.8)")
    ax.axhline(-0.8, color="red", lw=0.8, ls="--", alpha=0.5)

    ax.plot(layers, formality,  marker="o", lw=1.8, color="#1f77b4", label="formality")
    ax.plot(layers, confidence, marker="s", lw=1.8, color="#d62728", label="confidence")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine with self-reification direction")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"{label}: discriminant validity by layer")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    PAPER.mkdir(exist_ok=True)
    for label, (stem, best, n) in MODELS.items():
        slug = "llama" if "Llama" in label else "qwen"
        plot_model(label, stem, best, n, PAPER / f"figure_layerwise_discriminant_{slug}")


if __name__ == "__main__":
    main()
