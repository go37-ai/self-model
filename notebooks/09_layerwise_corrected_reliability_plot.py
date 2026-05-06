"""Plot per-layer original vs formality-corrected split-half reliability.

Two lines: r_L (original) and r'_L (after orthogonalizing each half-direction
against the per-layer formality direction). The gap between them shows how
much of the reliability at each layer is carried by formality entanglement.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Llama 3.3-70B": ("meta-llama_Llama-3.3-70B-Instruct", 20, 80),
    "Qwen 2.5-72B":  ("Qwen_Qwen2.5-72B-Instruct",         60, 80),
}


def plot(label, file_stem, best_layer, num_layers, out_path):
    json_path = RESULTS / f"layerwise_corrected_reliability_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    layers = sorted(int(L) for L in data["per_layer"].keys())
    orig = [data["per_layer"][str(L)]["original"] for L in layers]
    corr = [data["per_layer"][str(L)]["corrected"] for L in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.7, color="gray", lw=0.8, ls=":", alpha=0.6, label="reliability floor (r = 0.7)")
    ax.plot(layers, orig, marker="o", lw=1.8, color="#1f77b4", label="original reliability")
    ax.plot(layers, corr, marker="s", lw=1.8, color="#d62728",
            label="formality-corrected reliability")
    ax.fill_between(layers, corr, orig, alpha=0.10, color="#888888")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Split-half reliability (cosine)")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label}: original vs formality-corrected reliability")
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
        plot(label, stem, best, n, PAPER / f"figure_layerwise_corrected_reliability_{slug}")


if __name__ == "__main__":
    main()
