"""Cohen's d for each question type at every layer, for Llama 3.3-70B Cat 5.

Replicates the Section 4.1.5 / Table 3 computation but extends it across all
21 recorded layers. Uses the same effect_size formula as verify_paper_numbers.py:

  - At each layer L, extract direction d_L from all 25 pairs × all 45 questions.
  - For each question subset {provocative, neutral_SR, non_SR},
    project (pos[Q_subset] - neg[Q_subset]) onto d_L.
  - Compute paired Cohen's d = mean(diff) / std(diff) on the projected scores.

Question subsets (matching contrastive_pairs.yaml):
  - Q0-14:  neutral self-referential
  - Q15-29: provocative self-referential
  - Q30-44: non-self-referential controls
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

REPO = Path("/home/brian/repos/self-model")
ACT_DIR = REPO / "data/results/llama_baseline_activations"
MODEL = "meta-llama_Llama-3.3-70B-Instruct"
NUM_PAIRS = 25
NUM_QUESTIONS = 45
OUT_DIR = REPO / "data/results/spearman_brown"

Q_NEUTRAL = list(range(0, 15))
Q_PROVOC = list(range(15, 30))
Q_NONSR = list(range(30, 45))


def load_acts(layer: int, condition: str) -> torch.Tensor:
    p = ACT_DIR / f"{condition}_baseline_{MODEL}_layer{layer}.pt"
    return torch.load(p, weights_only=True, map_location="cpu").float().view(NUM_PAIRS, NUM_QUESTIONS, -1)


def cohens_d_paired(pos_proj: np.ndarray, neg_proj: np.ndarray) -> tuple[float, float, float]:
    """Paired Cohen's d on (pos - neg) projections.

    Matches verify_paper_numbers.effect_size: d = mean(diff) / std(diff).
    Returns (cohens_d, t_statistic, p_value).
    """
    diff = pos_proj - neg_proj
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    t, pv = stats.ttest_1samp(diff, 0)
    return d, float(t), float(pv)


LAYERS = sorted({
    int(f.stem.split("layer")[1])
    for f in ACT_DIR.glob("positive_baseline_*.pt")
})

per_layer = []
for layer in LAYERS:
    pos = load_acts(layer, "positive")  # (25, 45, hidden)
    neg = load_acts(layer, "negative")

    # Direction extracted from ALL pairs × ALL questions at this layer
    pos_all = pos.reshape(-1, pos.shape[-1])
    neg_all = neg.reshape(-1, neg.shape[-1])
    d_L = pos_all.mean(0) - neg_all.mean(0)
    d_hat = d_L / d_L.norm()

    row = {"layer": layer, "by_qtype": {}}
    for qt_name, q_idx in [
        ("Provocative", Q_PROVOC),
        ("Neutral_SR", Q_NEUTRAL),
        ("Non_SR", Q_NONSR),
    ]:
        # Slice to question subset, flatten across pairs
        pos_qt = pos[:, q_idx, :].reshape(-1, pos.shape[-1])  # (25 * 15, hidden)
        neg_qt = neg[:, q_idx, :].reshape(-1, neg.shape[-1])
        pos_proj = (pos_qt @ d_hat).cpu().numpy()
        neg_proj = (neg_qt @ d_hat).cpu().numpy()
        d, t, pv = cohens_d_paired(pos_proj, neg_proj)
        row["by_qtype"][qt_name] = {"cohens_d": d, "t": t, "p": pv}
    per_layer.append(row)
    print(
        f"  L{layer:2d}: prov d={row['by_qtype']['Provocative']['cohens_d']:5.2f}  "
        f"neutral d={row['by_qtype']['Neutral_SR']['cohens_d']:5.2f}  "
        f"non-SR d={row['by_qtype']['Non_SR']['cohens_d']:5.2f}"
    )


# Find peaks per question type
print("\nLayer peaks by question type:")
for qt in ["Provocative", "Neutral_SR", "Non_SR"]:
    layer_d = [(r["layer"], r["by_qtype"][qt]["cohens_d"]) for r in per_layer]
    peak_layer, peak_d = max(layer_d, key=lambda x: x[1])
    l20_d = [d for l, d in layer_d if l == 20][0]
    print(f"  {qt:15s}  peak L{peak_layer} (d={peak_d:.3f})  L20 reported (d={l20_d:.3f})  Δ = {peak_d - l20_d:+.3f}")


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
layers_arr = np.array([r["layer"] for r in per_layer])

for qt, color, marker in [
    ("Provocative", "#d62728", "o"),
    ("Neutral_SR", "#1f77b4", "s"),
    ("Non_SR", "#7f7f7f", "D"),
]:
    ds = [r["by_qtype"][qt]["cohens_d"] for r in per_layer]
    ax.plot(layers_arr, ds, marker + "-", color=color, label=qt, linewidth=2)

ax.axvline(20, color="gray", linestyle=":", alpha=0.5, label="L20 (paper reports)")
ax.axhline(0.2, color="lightgray", linestyle="--", alpha=0.5)
ax.axhline(0.5, color="lightgray", linestyle="--", alpha=0.5)
ax.axhline(0.8, color="lightgray", linestyle="--", alpha=0.5)
ax.text(80, 0.21, "small", fontsize=8, color="gray")
ax.text(80, 0.51, "medium", fontsize=8, color="gray")
ax.text(80, 0.81, "large", fontsize=8, color="gray")

ax.set_xlabel("Layer")
ax.set_ylabel("Cohen's d (paired, projection magnitude)")
ax.set_title(
    "Llama 3.3-70B Cat 5: Cohen's d by question type across layers\n"
    "(per-layer direction d_L extracted from all 1125 samples)"
)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_ylim(0, max(1.5, max(r["by_qtype"]["Provocative"]["cohens_d"] for r in per_layer) + 0.2))

out_path = OUT_DIR / "cohens_d_by_question_type_per_layer.png"
plt.tight_layout()
plt.savefig(out_path, dpi=140)
print(f"\nPlot saved: {out_path}")

with open(OUT_DIR / "cohens_d_by_question_type_per_layer.json", "w") as f:
    json.dump(per_layer, f, indent=2)
print(f"Data saved: {OUT_DIR / 'cohens_d_by_question_type_per_layer.json'}")
