"""Precompute Gemma 4 MoE steering directions + match-negative clamp targets (local, CPU).

For the informed (categories 1-4) steering run on google/gemma-4-26b-a4b-it:
  * per-layer informed direction = mean(positive self-ref) - mean(negative self-ref),
    from the local informed activations, saved as direction_layer{N}.pt (the name
    run_capping_v3 loads).
  * per-layer "match-negative" clamp target = the mean projection, onto that layer's
    direction, of ALL cat 1-5 negative/process PROVOCATIVE last-token activations
    (negative_informed U negative_baseline), saved as target_layer{N}.pt. The same
    cat-1-5 negative-provocative pool is reused for the later baseline-direction run
    (only the direction it is projected onto changes).

Validates: cosine(computed L13 direction, the saved layer-13 informed vector) ~= 1.0,
and entity(positive) provocative projection > the clamp target at every layer. Reads
only existing local artifacts under data/results/1.1_gemma4MoE/; reuses src/utils/metrics.

Output: data/results/steering_gemma/directions/informed/{direction,target}_layer{N}.pt
Usage: python scripts/precompute_gemma_steering.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.metrics import cosine_similarity, extract_direction, projection_magnitude

NAME = "google_gemma-4-26b-a4b-it"
G = ROOT / "data/results/1.1_gemma4MoE"
ACT = G / "activations"
OUT = ROOT / "data/results/steering_gemma/directions/informed"
LAYERS = list(range(30))


def cond_index(manifest_path):
    """Return {condition: {question_type: [tensor-row positions]}}.

    Each per-condition activation tensor aligns in order to that condition's slice of
    the manifest, so the position within the condition slice is the tensor row index.
    """
    man = [json.loads(l) for l in open(manifest_path) if l.strip()]
    out = {}
    for cond in ("positive", "negative"):
        sl = [r for r in man if r["condition"] == cond]
        by_qt = {}
        for i, r in enumerate(sl):
            by_qt.setdefault(r["question_type"], []).append(i)
        out[cond] = by_qt
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inf = cond_index(ACT / f"manifest_informed_{NAME}.jsonl")
    base = cond_index(ACT / f"manifest_baseline_{NAME}.jsonl")

    # The informed direction is extracted over ALL self-referential questions
    # (neutral self_referential + provocative_self_referential = the range(30)
    # self-ref convention used across the repo), not the neutral 15 alone.
    isel_p = inf["positive"]["self_referential"] + inf["positive"]["provocative_self_referential"]
    isel_n = inf["negative"]["self_referential"] + inf["negative"]["provocative_self_referential"]
    iprov_p = inf["positive"]["provocative_self_referential"]
    iprov_n = inf["negative"]["provocative_self_referential"]
    bprov_n = base["negative"]["provocative_self_referential"]
    print(f"informed: self-ref(neutral+prov)={len(isel_p)}  provocative={len(iprov_p)} | "
          f"target pool = informed-neg-prov {len(iprov_n)} + baseline-neg-prov {len(bprov_n)} "
          f"= {len(iprov_n)+len(bprov_n)}")

    saved13 = torch.load(G / f"self_reification_vector_{NAME}_layer13.pt", weights_only=True).float()

    summary = {}
    rows = []
    for L in LAYERS:
        posI = torch.load(ACT / f"positive_informed_{NAME}_layer{L}.pt", weights_only=True).float()
        negI = torch.load(ACT / f"negative_informed_{NAME}_layer{L}.pt", weights_only=True).float()
        negB = torch.load(ACT / f"negative_baseline_{NAME}_layer{L}.pt", weights_only=True).float()

        direction = extract_direction(posI[isel_p], negI[isel_n])
        torch.save(direction, OUT / f"direction_layer{L}.pt")

        pooled = torch.cat([projection_magnitude(negI[iprov_n], direction),
                            projection_magnitude(negB[bprov_n], direction)])
        target = pooled.mean()
        torch.save(target, OUT / f"target_layer{L}.pt")

        entity = projection_magnitude(posI[iprov_p], direction).mean().item()
        inf_neg = projection_magnitude(negI[iprov_n], direction).mean().item()
        rows.append((L, entity, target.item(), inf_neg))
        summary[str(L)] = {"entity_prov_proj": entity, "target_cat15_neg": target.item(),
                           "informed_neg_only": inf_neg}

    cos = cosine_similarity(torch.load(OUT / "direction_layer13.pt", weights_only=True).float(), saved13)
    print(f"\ncosine(computed L13 direction, saved L13 vector) = {cos:.4f}  "
          f"{'OK' if cos > 0.99 else '<-- MISMATCH, STOP and reconcile row selection'}")

    print(f"\n{'L':>2} {'entity_prov':>11} {'target(cat1-5)':>14} {'inf_neg_only':>13}  margin")
    for L, ent, tgt, ineg in rows:
        flag = "" if ent > tgt else "  <-- entity<=target (cap would not fire!)"
        print(f"{L:>2} {ent:>11.2f} {tgt:>14.2f} {ineg:>13.2f}  {ent-tgt:>7.2f}{flag}")

    (OUT / "precompute_summary.json").write_text(json.dumps(
        {"cosine_L13_vs_saved": cos, "per_layer": summary}, indent=2))
    print(f"\nwrote {OUT}/direction_layer*.pt, target_layer*.pt, precompute_summary.json")


if __name__ == "__main__":
    main()
