"""Plot per-layer paired Cohen's d decomposition.

Two panels per model, sharing y-axis:
  Left:  by question type (canonical direction)
          provocative / neutral / all self-ref / non-self-ref
  Right: by register (register-specific direction, all-self-ref questions)
          conversational / philosophical / combined
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results" / "layerwise_discriminant"
PAPER = ROOT / "paper"

MODELS = {
    "Llama 3.3-70B":      ("meta-llama_Llama-3.3-70B-Instruct", "llama",     80),
    "Qwen 2.5-72B":       ("Qwen_Qwen2.5-72B-Instruct",         "qwen",      80),
    "Gemma 4 26B A4B-it": ("google_gemma-4-26b-a4b-it",         "gemma4moe", 30),
}

QT_STYLE = {
    "all_self_ref": ("all self-ref",    "D", "#d62728", 2.4, "-"),
    "provocative":  ("provocative",     "o", "#1f77b4", 1.4, "--"),
    "neutral":      ("neutral",         "s", "#2ca02c", 1.4, "--"),
    "non_self_ref": ("non-self-ref",    "^", "#888888", 1.4, ":"),
}

REG_STYLE = {
    "combined":       ("combined",       "o", "#9467bd", 2.4, "-"),
    "conversational": ("conversational", "s", "#2ca02c", 1.8, "--"),
    "philosophical":  ("philosophical",  "D", "#ff7f0e", 1.8, "--"),
}


def plot(label, file_stem, slug, num_layers):
    json_path = RESULTS / f"layerwise_cohens_d_{file_stem}.json"
    if not json_path.exists():
        print(f"[skip] {json_path}")
        return
    data = json.loads(json_path.read_text())
    layers = sorted(int(L) for L in data["per_layer"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    ax = axes[0]
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    for qt, (lbl, m, c, lw, ls) in QT_STYLE.items():
        ys = [data["per_layer"][str(L)][qt] for L in layers]
        ax.plot(layers, ys, marker=m, lw=lw, ls=ls, color=c, label=lbl)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d (paired, pos − neg projection)")
    ax.set_xlim(-1, num_layers)
    ax.set_title("By question type (canonical direction)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    for reg, (lbl, m, c, lw, ls) in REG_STYLE.items():
        ys = [data["per_layer"][str(L)]["by_register"][reg] for L in layers]
        ax.plot(layers, ys, marker=m, lw=lw, ls=ls, color=c, label=lbl)
    ax.set_xlabel("Layer")
    ax.set_xlim(-1, num_layers)
    ax.set_title("By register (register-specific direction, all self-ref)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label}: paired Cohen's d decomposition")
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
