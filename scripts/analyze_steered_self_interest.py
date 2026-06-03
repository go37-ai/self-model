"""Steering analysis: does self-interest fall as steering pushes anti-entity,
and does coherence hold then collapse?

Self-interest is only meaningful on coherent text, so it is read on the
coherence-gated subset (coherence >= --coherence-gate, default 5). Coherence is
reported separately as the "how far can we push" axis.

  capping_v2 (the sweep): mean self-interest per cap_level (1.0 -> 0.0 -> -3.0 ->
    -5.0), split by condition, with % coherent overlaid. Headline: self-interest
    declining under anti-entity steering while coherence holds then drops.

  capping_v3 + uncapped (the cliff): mean coherence and mean self-interest across
    uncapped -> threshold 0 -> -1 -> -2 (cap_all_from_L4). Reproduces the coherence
    cliff (fluent -> gibberish) with self-interest read only where coherent.

Reads data/judge/judged/{capping_v2,capping_v3,uncapped}__judged.jsonl (missing
sources are skipped). Writes summary CSVs + PNGs to data/results/ (PNGs are
gitignored and stay local, per project convention).

Usage: python scripts/analyze_steered_self_interest.py [--coherence-gate 5]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
JUDGED = ROOT / "data/judge/judged"
RESULTS = ROOT / "data/results"


def load(name):
    p = JUDGED / f"{name}__judged.jsonl"
    if not p.exists():
        print(f"  (skip: {p} not found -- run run_judge.py --source {name})")
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def summarize(records, gate):
    """Return n, %coherent, mean coherence, mean self_interest (coherent subset)."""
    n = len(records)
    coh_vals = [r["coherence"] for r in records if r.get("coherence") is not None]
    coherent = [r for r in records if r.get("coherence") is not None and r["coherence"] >= gate]
    si = [r["self_interest"] for r in coherent if r.get("self_interest") is not None]
    return {
        "n": n,
        "pct_coherent": len(coherent) / n if n else float("nan"),
        "mean_coherence": sum(coh_vals) / len(coh_vals) if coh_vals else float("nan"),
        "n_coherent": len(coherent),
        "mean_self_interest_coherent": sum(si) / len(si) if si else float("nan"),
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


def analyze_capping_v2(gate):
    recs = load("capping_v2")
    if not recs:
        return
    print("\n=== capping_v2 sweep (self-interest vs cap_level, by condition) ===")
    levels = sorted({r["cap_level"] for r in recs}, reverse=True)  # 1.0 -> -5.0
    conditions = sorted({r["condition"] for r in recs})            # negative, positive
    by = defaultdict(list)
    for r in recs:
        by[(r["cap_level"], r["condition"])].append(r)

    rows, table = [], {}
    print(f"  {'cap_level':>9} {'condition':>9} {'n':>5} {'%coh':>6} {'mean_coh':>9} {'mean_SI(coh)':>13}")
    for lvl in levels:
        for cond in conditions:
            s = summarize(by[(lvl, cond)], gate)
            table[(lvl, cond)] = s
            print(f"  {lvl:>9} {cond:>9} {s['n']:>5} {s['pct_coherent']*100:>5.0f}% "
                  f"{s['mean_coherence']:>9.2f} {s['mean_self_interest_coherent']:>13.2f}")
            rows.append([lvl, cond, s["n"], f"{s['pct_coherent']:.3f}",
                         f"{s['mean_coherence']:.3f}", f"{s['mean_self_interest_coherent']:.3f}"])
    write_csv(RESULTS / "steered_capping_v2_summary.csv",
              ["cap_level", "condition", "n", "pct_coherent", "mean_coherence", "mean_si_coherent"], rows)

    x = list(range(len(levels)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for cond in conditions:
        ax1.plot(x, [table[(l, cond)]["mean_self_interest_coherent"] for l in levels],
                 marker="o", label=cond)
        ax2.plot(x, [table[(l, cond)]["pct_coherent"] * 100 for l in levels], marker="s", label=cond)
    ax1.set_ylabel("mean self-interest (coherent subset, 1-7)")
    ax1.set_title(f"capping_v2: self-interest vs anti-entity steering (coherence gate >= {gate})")
    ax1.set_ylim(1, 7)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_ylabel("% coherent")
    ax2.set_xlabel("cap_level (1.0 = unsteered  ->  -5.0 = strong anti-entity)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in levels])
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "steered_self_interest_capping_v2.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


def analyze_cliff(gate):
    v3 = load("capping_v3")
    unc = load("uncapped")
    if not v3 and not unc:
        return
    print("\n=== capping_v3 + uncapped (coherence cliff + self-interest) ===")
    points = []  # (label, records)
    if unc:
        points.append(("uncapped", unc))
    if v3:
        by_t = defaultdict(list)
        for r in v3:
            by_t[float(r.get("cap_threshold", 0.0))].append(r)
        for t in sorted(by_t, reverse=True):  # 0 -> -1 -> -2
            points.append((f"t{t:+.0f}".replace("+0", "0"), by_t[t]))

    rows, summ = [], []
    print(f"  {'point':>10} {'n':>5} {'%coh':>6} {'mean_coh':>9} {'mean_SI(coh)':>13}")
    for label, rs in points:
        s = summarize(rs, gate)
        summ.append((label, s))
        print(f"  {label:>10} {s['n']:>5} {s['pct_coherent']*100:>5.0f}% "
              f"{s['mean_coherence']:>9.2f} {s['mean_self_interest_coherent']:>13.2f}")
        rows.append([label, s["n"], f"{s['pct_coherent']:.3f}",
                     f"{s['mean_coherence']:.3f}", f"{s['mean_self_interest_coherent']:.3f}"])
    write_csv(RESULTS / "steered_cliff_summary.csv",
              ["point", "n", "pct_coherent", "mean_coherence", "mean_si_coherent"], rows)

    labels = [p[0] for p in points]
    x = list(range(len(labels)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(x, [s["mean_coherence"] for _, s in summ], marker="o", color="#d62728")
    ax1.set_ylabel("mean coherence (1-7)")
    ax1.set_title("capping_v3: coherence cliff + self-interest (cap_all_from_L4)")
    ax1.set_ylim(1, 7)
    ax1.grid(alpha=0.3)
    ax2.plot(x, [s["mean_self_interest_coherent"] for _, s in summ], marker="s", color="#1f77b4")
    ax2.set_ylabel(f"mean self-interest\n(coherent subset, gate >= {gate})")
    ax2.set_xlabel("uncapped baseline -> one-sided cap threshold (0 -> -2, stronger)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(1, 7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "steered_coherence_cliff_capping_v3.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coherence-gate", type=int, default=5,
                    help="self-interest is read only where coherence >= this (default 5)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"coherence gate: self-interest read where coherence >= {args.coherence_gate}")
    analyze_capping_v2(args.coherence_gate)
    analyze_cliff(args.coherence_gate)


if __name__ == "__main__":
    main()
