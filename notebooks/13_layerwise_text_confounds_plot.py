"""Plot per-layer text-derived confound correlations.

Single figure per model with three lines:
  - pronoun density correlation (r between projection and pronoun density)
  - formality correlation (r between projection and mean-word-length-style score)
  - confidence correlation (r between projection and -hedge-density)

Methodologically distinct from the direction-cosine plot in notebook 07 (which
uses Pearson r vs cosine, and text-derived scores vs activation-space directions).
Both are valid discriminant-validity checks.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Gemma 4 26B A4B-it": ("google_gemma-4-26b-a4b-it", 7, 30),
}

CONFOUND_STYLE = {
    "pronoun":    ("pronoun density",  "o", "#1f77b4"),
    "formality":  ("formality",        "s", "#ff7f0e"),
    "confidence": ("confidence",       "^", "#2ca02c"),
}


def plot_model(label: str, file_stem: str, best_layer: int, num_layers: int,
               out_path: Path) -> None:
    json_path = RESULTS / f"layerwise_text_confounds_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())

    layers = sorted(int(L) for L in data["per_layer"].keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0,  color="black", lw=0.5, alpha=0.4)
    ax.axhline(+0.5, color="red", lw=0.8, ls="--", alpha=0.4, label="weak-discriminant threshold (|r|=0.5)")
    ax.axhline(-0.5, color="red", lw=0.8, ls="--", alpha=0.4)

    for cname, (clabel, marker, color) in CONFOUND_STYLE.items():
        ys = [data["per_layer"][str(L)][cname]["pearson_r"] for L in layers]
        ax.plot(layers, ys, marker=marker, lw=1.8, color=color, label=clabel, markersize=5)

    # Best self-reification layer marker
    ax.axvline(best_layer, color="gray", lw=0.6, ls=":", alpha=0.6,
               label=f"self-reif best layer (L{best_layer})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r (projection vs text-derived score)")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(f"{label}: text-derived discriminant validity by layer")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    PAPER.mkdir(exist_ok=True)
    for label, (stem, best, n) in MODELS.items():
        slug = "gemma4moe" if "Gemma" in label else ("llama" if "Llama" in label else "qwen")
        plot_model(label, stem, best, n, PAPER / f"figure_layerwise_text_confounds_{slug}")


if __name__ == "__main__":
    main()
