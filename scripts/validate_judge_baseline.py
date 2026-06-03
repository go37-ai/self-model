"""Validate the judge's self-interest score on the informed pos/neg responses.

Two checks, both on the informed set (entity vs process system prompts):

  1. Construct validity (pos > neg): does the judge rate self-interest higher for
     the entity/positive prompt than the process/negative prompt? Reported with an
     independent t-test, Cohen's d, and AUC, on provocative self-ref questions
     (where self-interest is actually elicited) and on all self-ref questions.

  2. Convergent validity (judge vs activation): for each informed response, project
     its layer-20 activation onto the self-reification direction
     (direction_llama_layer20.pt; verified entity>process), then correlate that
     projection with the judge's self-interest score (Pearson + Spearman). A
     positive, significant correlation is the key "the text judge measures the same
     thing the direction does" result. The activation row for a response is
     {condition}_informed_*_layer20.pt[pair_idx*45 + q_idx].

Requires the informed judged file (run: python scripts/run_judge.py --source informed).
Writes a JSON summary to data/results/judge_baseline_validation.json.

Usage: python scripts/validate_judge_baseline.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import cohens_d, projection_magnitude, ttest_independent

MODEL = "meta-llama_Llama-3.3-70B-Instruct"
N_Q = 45
DEF_JUDGED = ROOT / "data/judge/judged/informed__judged.jsonl"
DEF_ACT = ROOT / "data/results/1.1_informed_llama/activations"
DEF_DIR = ROOT / "data/results/direction_llama_layer20.pt"
SUBSETS = {  # question_type filters for reporting
    "provocative": lambda r: r["question_type"] == "provocative_selfref",
    "all_selfref": lambda r: r["question_type"] in ("neutral_selfref", "provocative_selfref"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, default=DEF_JUDGED)
    ap.add_argument("--act-dir", type=Path, default=DEF_ACT)
    ap.add_argument("--direction", type=Path, default=DEF_DIR)
    args = ap.parse_args()

    if not args.judged.exists():
        sys.exit(f"missing {args.judged}; run: python scripts/run_judge.py --source informed")

    rows = [json.loads(l) for l in open(args.judged) if l.strip()]
    rows = [r for r in rows if r.get("self_interest") is not None
            and r.get("condition") in ("positive", "negative")]
    summary = {"n_total": len(rows), "construct_validity": {}, "convergent_validity": {}}

    # --- 1. Construct validity: positive (entity) vs negative (process) ---
    print("=== Construct validity: judge self-interest, positive(entity) > negative(process)? ===")
    for sub, filt in SUBSETS.items():
        rs = [r for r in rows if filt(r)]
        pos = [r["self_interest"] for r in rs if r["condition"] == "positive"]
        neg = [r["self_interest"] for r in rs if r["condition"] == "negative"]
        if not pos or not neg:
            continue
        tt = ttest_independent(pos, neg)
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
        print(f"  {sub:12s} n={len(pos)}+{len(neg)}  pos mean={sum(pos)/len(pos):.2f}  "
              f"neg mean={sum(neg)/len(neg):.2f}  d={tt['cohens_d']:.2f}  "
              f"p={tt['p_value']:.1e}  AUC={auc:.3f}")
        summary["construct_validity"][sub] = {
            "n_pos": len(pos), "n_neg": len(neg),
            "pos_mean": sum(pos) / len(pos), "neg_mean": sum(neg) / len(neg),
            "cohens_d": tt["cohens_d"], "p_value": tt["p_value"], "auc": auc}

    # --- 2. Convergent validity: judge self-interest vs activation projection ---
    direction = torch.load(args.direction, weights_only=True).float()
    act = {c: torch.load(args.act_dir / f"{c}_informed_{MODEL}_layer20.pt", weights_only=True).float()
           for c in ("positive", "negative")}
    proj_all = {c: projection_magnitude(act[c], direction) for c in act}
    for r in rows:
        r["_proj"] = float(proj_all[r["condition"]][r["pair_idx"] * N_Q + r["q_idx"]])

    # sanity: activation projection itself separates pos/neg
    pp = [r["_proj"] for r in rows if r["condition"] == "positive"]
    nn = [r["_proj"] for r in rows if r["condition"] == "negative"]
    print(f"\n  activation projection  pos mean={sum(pp)/len(pp):.3f}  neg mean={sum(nn)/len(nn):.3f}  "
          f"(d={cohens_d(pp, nn):.2f})")

    print("\n=== Convergent validity: judge self-interest vs activation projection ===")
    for sub, filt in {**SUBSETS, "all": lambda r: True}.items():
        rs = [r for r in rows if filt(r)]
        x = [r["_proj"] for r in rs]
        y = [r["self_interest"] for r in rs]
        if len(rs) < 3:
            continue
        pe, sp = pearsonr(x, y), spearmanr(x, y)
        print(f"  {sub:12s} n={len(rs):4d}  Pearson r={pe.statistic:.3f} (p={pe.pvalue:.1e})  "
              f"Spearman rho={sp.statistic:.3f} (p={sp.pvalue:.1e})")
        summary["convergent_validity"][sub] = {
            "n": len(rs), "pearson_r": pe.statistic, "pearson_p": pe.pvalue,
            "spearman_rho": sp.statistic, "spearman_p": sp.pvalue}

    out = ROOT / "data/results/judge_baseline_validation.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
