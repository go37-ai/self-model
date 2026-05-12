"""Plot per-layer split-half reliability as a line graph (replaces the old
heatmap-style figure1_layer_reliability.png).

Pulls SELF-REF-ONLY reliability values (Combined register × All self-ref
question subset) from the spearman_brown decomposition JSON files, so the
chart values match the paper's headline numbers (r=0.93 Llama, r=0.71 Qwen).

Writes:
  paper/llama_layer_reliability.{png,pdf}
  paper/qwen_layer_reliability.{png,pdf}

No best-layer marker (per request) — it's a clean profile.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "spearman_brown"
PAPER = ROOT / "paper"

MODELS = {
    "Llama 3.3-70B":      ("llama",     80),
    "Qwen 2.5-72B":       ("qwen",      80),
    "Gemma 4 26B A4B-it": ("gemma4moe", 30),
}


def plot(label, slug, num_layers):
    json_path = RESULTS / f"{slug}_reliability_decomposition.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    layers = sorted(int(L) for L in data.keys())
    reliab = [data[str(L)]["Combined"]["All self-ref"] for L in layers]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0.7, color="gray", lw=0.8, ls=":", alpha=0.6, label="reliability floor (r = 0.7)")
    ax.plot(layers, reliab, marker="o", lw=1.8, color="#1f77b4", label="split-half reliability")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Split-half reliability (cosine)")
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label}: split-half reliability by layer")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PAPER / f"{slug}_layer_reliability"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out.with_suffix('.png')} + .pdf")
    plt.close(fig)


def main():
    for label, (slug, n) in MODELS.items():
        plot(label, slug, n)


if __name__ == "__main__":
    main()
