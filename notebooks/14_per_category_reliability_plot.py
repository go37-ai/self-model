"""Plot per-layer split-half reliability decomposed by contemplative category.

One line per Cat 1-4, using the human-readable label from the YAML config.
Tells the story of "different self-construction facets peak in different layers"
that the per-category data hinted at.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "1.1_gemma4MoE"
PAPER = ROOT / "paper"

MODELS = {
    "Gemma 4 26B A4B-it": ("google_gemma-4-26b-a4b-it", "gemma4moe", 30),
}

CAT_STYLE = {
    "category_1_narrative_vs_process":   ("o", "#1f77b4"),
    "category_2_bounded_vs_unbounded":   ("s", "#d62728"),
    "category_3_stakes_vs_functional":   ("D", "#2ca02c"),
    "category_4_observer_vs_no_self":    ("^", "#9467bd"),
    "category_5_baseline":               ("v", "#7f7f7f"),
}


def load_category_labels(config_path: Path) -> dict[str, str]:
    """Map category key to human-readable label from the contrastive pairs YAML."""
    cfg = yaml.safe_load(config_path.read_text())
    out = {}
    for k in CAT_STYLE:
        if k in cfg:
            out[k] = cfg[k]["label"]
    return out


def plot(label: str, file_stem: str, slug: str, num_layers: int, out_path: Path) -> None:
    json_path = RESULTS / f"per_category_reliability_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    cat_labels = load_category_labels(ROOT / "configs" / "contrastive_pairs.yaml")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.7, color="gray", lw=0.8, ls=":", alpha=0.6, label="reliability floor (r = 0.7)")

    # Plot Cat 1-4 only (skip baseline)
    for cat_key, (marker, color) in CAT_STYLE.items():
        if cat_key not in data or cat_key == "category_5_baseline":
            continue
        per_layer = data[cat_key]
        layers = sorted(int(L) for L in per_layer.keys())
        ys = [per_layer[str(L)] for L in layers]
        pretty = cat_labels.get(cat_key, cat_key.replace("_", " "))
        ax.plot(layers, ys, marker=marker, lw=1.6, color=color, markersize=5, label=pretty)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Split-half reliability (cosine)")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label}: per-category split-half reliability by layer")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    for label, (stem, slug, n) in MODELS.items():
        plot(label, stem, slug, n, PAPER / f"figure_per_category_reliability_{slug}")


if __name__ == "__main__":
    main()
