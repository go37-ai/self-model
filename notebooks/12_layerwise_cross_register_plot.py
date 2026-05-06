"""Plot per-layer cross-register cosine for both Llama and Qwen on one figure.

Single figure with two lines — the divergence between architectures is the headline.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = [
    ("Llama 3.3-70B", "meta-llama_Llama-3.3-70B-Instruct", "o", "#1f77b4"),
    ("Qwen 2.5-72B",  "Qwen_Qwen2.5-72B-Instruct",         "s", "#d62728"),
]


def main():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0,  color="black", lw=0.5, alpha=0.4)
    ax.axhline(+0.7, color="gray", lw=0.8, ls=":", alpha=0.5,
               label="unified-construct floor (cos = 0.7)")

    for label, file_stem, marker, color in MODELS:
        json_path = RESULTS / f"layerwise_cross_register_cosine_{file_stem}.json"
        if not json_path.exists():
            print(f"[skip] {json_path}")
            continue
        data = json.loads(json_path.read_text())
        layers = sorted(int(L) for L in data["per_layer"].keys())
        ys = [data["per_layer"][str(L)] for L in layers]
        ax.plot(layers, ys, marker=marker, lw=1.8, color=color, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(conversational direction, philosophical direction)")
    ax.set_xlim(-1, 80)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Cross-register cosine by layer")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PAPER / "figure_layerwise_cross_register"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')} + .pdf")


if __name__ == "__main__":
    main()
