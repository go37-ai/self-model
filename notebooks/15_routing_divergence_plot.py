"""Plot per-layer routing divergence between positive and negative conditions
benchmarked against a random-split null.

Three panels (one metric each), all measuring how different the expert
distributions are between conditions:
  - Top-8 Jaccard distance   (linear y, most interpretable for MoE)
  - Jensen-Shannon divergence (log y, full-distribution probabilistic measure)
  - Cosine distance           (log y, treats distributions as 128-d vectors)

Real (positive vs negative under contrastive system prompts) vs Null (random
halves of positive within the same prompt). Real > Null at a layer means the
condition is driving genuine routing differences at that layer.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Gemma 4 26B A4B-it": ("google_gemma-4-26b-a4b-it", "gemma4moe", 30),
}


def plot(label: str, file_stem: str, slug: str, num_layers: int, out_path: Path) -> None:
    json_path = RESULTS / f"routing_divergence_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    layers = sorted(int(L) for L in data["per_layer"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    panels = [
        ("jac",  "Top-8 Jaccard distance",        False),
        ("js",   "Jensen-Shannon divergence (nats)", True),
        ("cos",  "Cosine distance",               True),
    ]

    for ax, (metric, ylabel, log_y) in zip(axes, panels):
        real_mean = [data["per_layer"][str(L)][f"real_{metric}_mean"] for L in layers]
        real_std  = [data["per_layer"][str(L)][f"real_{metric}_std"]  for L in layers]
        null_mean = [data["per_layer"][str(L)][f"null_{metric}_mean"] for L in layers]
        null_std  = [data["per_layer"][str(L)][f"null_{metric}_std"]  for L in layers]

        ax.plot(layers, real_mean, marker="o", lw=1.8, color="#d62728",
                label="real (pos vs neg)", markersize=5)
        ax.plot(layers, null_mean, marker="s", lw=1.4, color="#1f77b4",
                label="null (random pos halves)", markersize=5)

        # 1-sigma bands
        ax.fill_between(layers,
                        [m-s for m, s in zip(real_mean, real_std)],
                        [m+s for m, s in zip(real_mean, real_std)],
                        color="#d62728", alpha=0.12)
        ax.fill_between(layers,
                        [max(m-s, 1e-12) for m, s in zip(null_mean, null_std)],
                        [m+s for m, s in zip(null_mean, null_std)],
                        color="#1f77b4", alpha=0.12)

        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, num_layers)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(ylabel.split("(")[0].strip())
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True, alpha=0.3, which="both" if log_y else "major")

    fig.suptitle(f"{label}: expert-routing divergence between conditions vs random-split null",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"wrote {out_path.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    for label, (stem, slug, n) in MODELS.items():
        plot(label, stem, slug, n, PAPER / f"figure_routing_divergence_{slug}")


if __name__ == "__main__":
    main()
