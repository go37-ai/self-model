"""Plot per-layer Pearson correlation between self-reification projection
and first-person pronoun density (kept separate from the cosine-based
formality/confidence figure since the two measures aren't comparable).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Llama 3.3-70B":      ("meta-llama_Llama-3.3-70B-Instruct", "llama",     20, 80),
    "Gemma 4 26B A4B-it": ("google_gemma-4-26b-a4b-it",         "gemma4moe", 7,  30),
}


def plot(label, file_stem, best_layer, num_layers, out_path):
    json_path = RESULTS / f"layerwise_pronoun_correlation_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    with open(json_path) as f:
        data = json.load(f)
    layers = sorted(int(L) for L in data["per_layer"].keys())
    rs = [data["per_layer"][str(L)]["pearson_r"] for L in layers]
    n = data["n_samples"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0.0,  color="black", lw=0.5, alpha=0.4)
    ax.axhline(+0.8, color="red", lw=0.8, ls="--", alpha=0.5, label="discriminant threshold (±0.8)")
    ax.axhline(-0.8, color="red", lw=0.8, ls="--", alpha=0.5)
    ax.plot(layers, rs, marker="^", lw=1.8, color="#2ca02c",
            label=f"pronoun density (Pearson r, n={n})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r (projection vs pronoun density)")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"{label}: pronoun-density correlation by layer")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    PAPER.mkdir(exist_ok=True)
    for label, (stem, slug, best, n) in MODELS.items():
        plot(label, stem, best, n, PAPER / f"figure_layerwise_pronoun_{slug}")


if __name__ == "__main__":
    main()
