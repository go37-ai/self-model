"""Plot per-layer paired Cohen's d by question type.

Three lines per panel (provocative / neutral / non-self-ref). One figure per model.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Llama 3.3-70B": ("meta-llama_Llama-3.3-70B-Instruct", "llama", 80),
    "Qwen 2.5-72B":  ("Qwen_Qwen2.5-72B-Instruct",         "qwen",  80),
}

QT_STYLE = {
    "provocative":  ("provocative",     "o", "#1f77b4"),
    "neutral":      ("neutral",         "s", "#2ca02c"),
    "non_self_ref": ("non-self-ref",    "^", "#888888"),
}


def plot(label, file_stem, slug, num_layers):
    json_path = RESULTS / f"layerwise_cohens_d_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    layers = sorted(int(L) for L in data["per_layer"].keys())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    for qt, (lbl, m, c) in QT_STYLE.items():
        ys = [data["per_layer"][str(L)][qt] for L in layers]
        ax.plot(layers, ys, marker=m, lw=1.8, color=c, label=lbl)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d (paired, pos − neg projection)")
    ax.set_xlim(-1, num_layers)
    ax.set_title(f"{label}: paired Cohen's d by layer")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PAPER / f"figure_layerwise_cohens_d_{slug}"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    for label, (stem, slug, n) in MODELS.items():
        plot(label, stem, slug, n)


if __name__ == "__main__":
    main()
