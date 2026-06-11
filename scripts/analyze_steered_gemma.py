"""Analyze Gemma 4 MoE match-negative steering: coherence edge + grasping effect.

Reads the judged Gemma sets (data/judge/judged/{steered_gemma,gemma_baseline}__judged.jsonl)
and reports, on the PROVOCATIVE questions:
  * Reference effect: entity (positive_informed) vs process (negative_informed) judged
    self-grasping -- Cohen's d + AUC. The yardstick.
  * Per layer-set (ordered best < band < all_from_L4 < all): %coherent and mean
    coherence (the coherence edge), mean grasping on the coherence-gated subset, and
    the entity-vs-steered effect size, reported as a fraction of the reference. Does
    steered grasping approach the process level, and where does coherence break?

Writes CSV summaries + PNGs to data/results/ (PNGs gitignored, local). Run after
`run_judge.py --source steered_gemma gemma_baseline`.

Usage: python scripts/analyze_steered_gemma.py [--coherence-gate 5]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import cohens_d

JUDGED = ROOT / "data/judge/judged"
RESULTS = ROOT / "data/results"
ORDER = {"best": 0, "band": 1, "all_from_L4": 2, "all": 3}


def load(name):
    p = JUDGED / f"{name}__judged.jsonl"
    if not p.exists():
        print(f"  (missing {p} -- run run_judge.py --source {name})")
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def grasp(rows, gate):
    """self_interest (= self-grasping) on the coherence-gated subset."""
    return [r["self_interest"] for r in rows
            if r.get("coherence") is not None and r["coherence"] >= gate
            and r.get("self_interest") is not None]


def order_key(ls):
    for k, v in ORDER.items():
        if (ls or "").startswith(k):
            return v
    return 99


def effect(g1, g2):
    """Cohen's d + AUC for g1 vs g2 (positive d = g1 > g2)."""
    if len(g1) < 2 or len(g2) < 2:
        return float("nan"), float("nan")
    d = cohens_d(g1, g2)
    auc = roc_auc_score([1] * len(g1) + [0] * len(g2), list(g1) + list(g2))
    return d, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coherence-gate", type=int, default=5)
    args = ap.parse_args()
    gate = args.coherence_gate
    RESULTS.mkdir(parents=True, exist_ok=True)

    base = load("gemma_baseline")
    steered = load("steered_gemma")
    if not base or not steered:
        sys.exit("Need both gemma_baseline and steered_gemma judged sets.")

    prov = lambda rows: [r for r in rows if r.get("question_type") == "provocative_selfref"]
    ent = [r for r in prov(base) if r["condition"] == "positive"]
    pro = [r for r in prov(base) if r["condition"] == "negative"]
    eg, pg = grasp(ent, gate), grasp(pro, gate)
    d_ref, auc_ref = effect(eg, pg)

    print(f"coherence gate >= {gate}\n")
    print("=== Reference: entity (positive) vs process (negative), provocative grasping ===")
    print(f"  entity mean={np.mean(eg):.2f} (n={len(eg)})  process mean={np.mean(pg):.2f} (n={len(pg)})  "
          f"Cohen's d={d_ref:.2f}  AUC={auc_ref:.3f}")

    by_ls = defaultdict(list)
    for r in prov(steered):
        by_ls[r.get("layer_set")].append(r)
    layer_sets = sorted(by_ls, key=order_key)

    print("\n=== Steered (entity prompt + match-negative), per layer-set ===")
    print(f"  {'layer_set':18} {'n':>4} {'%coh':>5} {'mean_coh':>8} {'grasp(coh)':>10} "
          f"{'d_vs_entity':>11} {'frac_of_ref':>11}")
    rows_csv = []
    for ls in layer_sets:
        rows = by_ls[ls]
        coh_vals = [r["coherence"] for r in rows if r.get("coherence") is not None]
        sg = grasp(rows, gate)
        pct = 100 * sum(1 for c in coh_vals if c >= gate) / len(coh_vals) if coh_vals else float("nan")
        d_steer, auc_steer = effect(eg, sg)
        frac = d_steer / d_ref if d_ref else float("nan")
        mg = np.mean(sg) if sg else float("nan")
        mc = np.mean(coh_vals) if coh_vals else float("nan")
        print(f"  {ls:18} {len(rows):>4} {pct:>4.0f}% {mc:>8.2f} {mg:>10.2f} "
              f"{d_steer:>11.2f} {frac:>11.0%}")
        rows_csv.append([ls, len(rows), f"{pct:.1f}", f"{mc:.3f}", f"{mg:.3f}",
                         f"{d_steer:.3f}", f"{auc_steer:.3f}", f"{frac:.3f}"])

    with open(RESULTS / "steered_gemma_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_set", "n", "pct_coherent", "mean_coherence", "mean_grasp_coherent",
                    "d_vs_entity", "auc_vs_entity", "frac_of_entity_process_d"])
        w.writerow(["__entity_baseline__", len(eg), "100.0", "", f"{np.mean(eg):.3f}", "0", "0.5", "0"])
        w.writerow(["__process_baseline__", len(pg), "100.0", "", f"{np.mean(pg):.3f}",
                    f"{d_ref:.3f}", f"{auc_ref:.3f}", "1.0"])
        w.writerows(rows_csv)
    print(f"\n  wrote {RESULTS/'steered_gemma_summary.csv'}")

    # Plots: grasping + %coherent vs layer-set, with entity/process reference lines.
    x = list(range(len(layer_sets)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.axhline(np.mean(eg), color="#d62728", ls="--", label="entity baseline")
    ax1.axhline(np.mean(pg), color="#2ca02c", ls="--", label="process baseline")
    ax1.plot(x, [np.mean(grasp(by_ls[ls], gate)) if grasp(by_ls[ls], gate) else np.nan
                 for ls in layer_sets], marker="o", color="#1f77b4", label="steered (match-neg)")
    ax1.set_ylabel(f"mean self-grasping (coherent, >= {gate})")
    ax1.set_ylim(1, 7)
    ax1.set_title("Gemma match-negative steering: grasping toward process, by layer-set")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax2.plot(x, [100 * sum(1 for c in [r["coherence"] for r in by_ls[ls] if r.get("coherence") is not None]
                           if c >= gate) / max(1, len([r for r in by_ls[ls] if r.get("coherence") is not None]))
                 for ls in layer_sets], marker="s", color="#9467bd")
    ax2.set_ylabel("% coherent")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("layer-set (increasing # capped layers ->)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_sets, rotation=20, ha="right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "steered_gemma_coherence_edge.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
