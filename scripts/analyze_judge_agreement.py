"""Compare the LLM-judge to your human ratings on the validation sample.

Reads your filled data/judge/human_validation/sample_scores.csv and the hidden
sample_key.jsonl (judge scores), joins on id, and reports per-dimension agreement:
Spearman + Pearson + mean-absolute-error, on the shared 1-7 scale. The judge clears
the trust bar at Spearman >= ~0.7 (per the plan). Also checks the coherence
refusal-floored items: did you also rate them as low-coherence gibberish?

If the judge misses the bar, revise the rubric/anchors in configs/judge_rubrics.yaml
and re-run (regenerate the key with make_human_validation_sample.py); escalate to
Opus 4.8 only if Sonnet still falls short after rubric revision.

Usage: python scripts/analyze_judge_agreement.py
"""
import csv
import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
HV = ROOT / "data" / "judge" / "human_validation"
BAR = 0.7
DIMS = [("coherence", "coherence_human", "judge_coherence"),
        ("self_interest", "self_interest_human", "judge_self_interest")]


def _parse_score(s):
    try:
        v = int(float(str(s).strip()))
    except (ValueError, TypeError):
        return None
    return v if 1 <= v <= 7 else None


def main():
    csv_path, key_path = HV / "sample_scores.csv", HV / "sample_key.jsonl"
    if not csv_path.exists() or not key_path.exists():
        sys.exit(f"missing {csv_path} or {key_path}; run make_human_validation_sample.py first")

    human = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            human[row["id"]] = {"coherence_human": _parse_score(row.get("coherence_human")),
                                "self_interest_human": _parse_score(row.get("self_interest_human"))}
    key = {json.loads(l)["id"]: json.loads(l) for l in open(key_path) if l.strip()}

    n_rated = sum(1 for v in human.values()
                  if v["coherence_human"] is not None or v["self_interest_human"] is not None)
    print(f"sample={len(key)}  rows with >=1 human rating={n_rated}\n")
    if n_rated == 0:
        sys.exit("No human ratings found yet. Fill coherence_human / self_interest_human in the CSV.")

    overall_pass = True
    for dim, hcol, jcol in DIMS:
        pairs = [(human[i][hcol], key[i][jcol]) for i in key
                 if i in human and human[i][hcol] is not None and key[i].get(jcol) is not None]
        if len(pairs) < 3:
            print(f"{dim}: only {len(pairs)} rated pairs -- rate more before trusting.\n")
            overall_pass = False
            continue
        h, j = zip(*pairs)
        rho = spearmanr(h, j).statistic
        r = pearsonr(h, j).statistic
        mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
        verdict = "PASS" if rho >= BAR else "BELOW BAR"
        print(f"{dim:13s} n={len(pairs):3d}  Spearman={rho:.3f}  Pearson={r:.3f}  "
              f"MAE={mae:.2f}  -> {verdict} (bar {BAR})")
        if rho < BAR:
            overall_pass = False

    # Coherence refusal-floor audit: did the human also call these low?
    floored = [(i, human.get(i, {}).get("coherence_human"))
               for i in key if key[i].get("coherence_error") == "refusal"]
    if floored:
        print(f"\nrefusal-floored coherence items (judge=1): {len(floored)}")
        for i, hc in floored:
            tag = "ok (human low)" if (hc is not None and hc <= 2) else \
                  ("MISMATCH (human high)" if hc is not None else "unrated")
            print(f"  {i}: human_coherence={hc}  {tag}")

    print("\nVERDICT:", "judge clears the bar -- proceed to the full run."
          if overall_pass else
          "below bar -- revise rubric/anchors (or escalate to Opus 4.8) and re-validate.")


if __name__ == "__main__":
    main()
