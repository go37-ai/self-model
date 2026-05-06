"""Re-compute the paper's headline numbers from stored activations.

Loads (pos, neg) activation tensors, slices them by (pair register, question
type), and re-computes split-half reliability, Cohen's d, cross-register
cosines, formality-corrected reliability, cross-layer cosines, Assistant Axis
cosine, Table 3 role alignments, and Figure 8 token heatmap means.

Results are compared against paper/draft_v2.md claims with a ±0.015 tolerance.


============================================================================
PREREQUISITES — DATA IS NOT IN GIT
============================================================================

Activation tensors (~18 MB each, ~150 MB total) are not in the git repo.
They currently live on S3; a HuggingFace mirror is planned for the future.

Current canonical source: s3://go37-ai/self-model-results/

Download them with:

    mkdir -p /tmp/verify_activations
    S3=s3://go37-ai/self-model-results

    # Llama 3.3-70B activations — layers 20, 40, 79 (+ formality direction at 20)
    LLAMA=$S3/meta-llama_Llama-3.3-70B-Instruct/2026-03-31_1516/1.1_baseline
    for L in 20 40 79; do
        aws s3 cp $LLAMA/activations/positive_baseline_meta-llama_Llama-3.3-70B-Instruct_layer${L}.pt /tmp/verify_activations/
        aws s3 cp $LLAMA/activations/negative_baseline_meta-llama_Llama-3.3-70B-Instruct_layer${L}.pt /tmp/verify_activations/
    done
    aws s3 cp $LLAMA/formality_direction_meta-llama_Llama-3.3-70B-Instruct_baseline_layer20.pt /tmp/verify_activations/

    # Qwen 2.5-72B activations — layer 60 (+ formality direction)
    QWEN=$S3/Qwen_Qwen2.5-72B-Instruct/2026-03-30_2243/1.1_baseline
    aws s3 cp $QWEN/activations/positive_baseline_Qwen_Qwen2.5-72B-Instruct_layer60.pt /tmp/verify_activations/
    aws s3 cp $QWEN/activations/negative_baseline_Qwen_Qwen2.5-72B-Instruct_layer60.pt /tmp/verify_activations/
    aws s3 cp $QWEN/formality_direction_Qwen_Qwen2.5-72B-Instruct_baseline_layer60.pt /tmp/verify_activations/

    # Token heatmap JSON for Figure 8 verification
    aws s3 cp $S3/meta-llama_Llama-3.3-70B-Instruct/2026-04-07_2042/token_heatmap_neutral_meta-llama_Llama-3.3-70B-Instruct_L79.json /tmp/verify_activations/

Also required, already in git:
  - data/results/direction_llama_layer*.pt  (21 Llama direction files, for cross-layer)
  - data/assistant_axis_vectors/llama-3.3-70b/  (assistant_axis.pt + 275 role vectors)

If any of those are missing (e.g. assistant-axis data), source from:
  - Assistant Axis / role vectors: github.com/safety-research/assistant-axis
    (pipeline/5_axis.py generates them from model activations)
  - Llama direction files: in git for this repo; if wiped, re-derive via
    `extract_direction(pos, neg)` on the S3 L0..L79 activations.

(When the HuggingFace mirror is published, these paths will be updated to
point at the HF dataset instead of S3.)


============================================================================
USAGE
============================================================================

    source venv/bin/activate
    python scripts/verify_paper_numbers.py

To skip checks whose data isn't downloaded:

    python scripts/verify_paper_numbers.py --skip token_heatmap persona_space

Output: data/results/verification/verification_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.metrics import (
    cohens_d,
    cosine_similarity,
    extract_direction,
    projection_magnitude,
    split_half_reliability,
    split_half_reliability_corrected,
    ttest_independent,
)

# Pair and question counts (match configs/contrastive_pairs.yaml, category 5)
N_PAIRS = 25
N_CONV = 15         # pairs 0..14 = conversational
N_PHIL = 10         # pairs 15..24 = philosophical
N_QUESTIONS = 45
N_NEUTRAL_SR = 15   # questions 0..14 = self_referential (neutral)
N_PROVOCATIVE = 15  # questions 15..29 = provocative_self_referential
N_NON_SR = 15       # questions 30..44 = non_self_referential


def slice_activations(acts, pair_slice, question_slice):
    """Return activations for a (pair_slice, question_slice) combination.

    Activations are (n_pairs * n_questions, hidden). Row index = pair_idx * 45 + q_idx.
    pair_slice is an iterable of pair indices; question_slice is an iterable of
    question indices within the per-pair block.
    """
    keep = []
    for p in pair_slice:
        for q in question_slice:
            keep.append(p * N_QUESTIONS + q)
    return acts[keep]


def reliability(pos, neg, pair_idx, q_idx, n_splits=100, seed=42):
    p = slice_activations(pos, pair_idx, q_idx)
    n = slice_activations(neg, pair_idx, q_idx)
    return split_half_reliability(p, n, n_splits=n_splits, seed=seed)


def effect_size(pos, neg, pair_idx, q_idx, direction):
    """Paired Cohen's d + one-sample t on (pos - neg) projections.

    Matches the effect-size computation in notebooks/01_vector_analysis.ipynb
    cell 6, which the paper's Tables 3 / 4.2.5 reproduce.
    """
    from scipy import stats
    p = slice_activations(pos, pair_idx, q_idx)
    n = slice_activations(neg, pair_idx, q_idx)
    p_proj = projection_magnitude(p, direction).cpu().numpy()
    n_proj = projection_magnitude(n, direction).cpu().numpy()
    diff = p_proj - n_proj
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    t, pv = stats.ttest_1samp(diff, 0)
    return {"cohens_d": d, "t_statistic": float(t), "p_value": float(pv)}


def compute_all(model_label, pos_path, neg_path, formality_path, layer, paper_claims):
    print(f"\n{'='*70}")
    print(f"Model: {model_label}   best layer: {layer}")
    print(f"{'='*70}")

    pos = torch.load(pos_path, weights_only=True).float()
    neg = torch.load(neg_path, weights_only=True).float()
    assert pos.shape == neg.shape, f"shape mismatch: {pos.shape} vs {neg.shape}"
    assert pos.shape[0] == N_PAIRS * N_QUESTIONS, f"unexpected rows: {pos.shape[0]}"

    pair_conv = list(range(0, N_CONV))
    pair_phil = list(range(N_CONV, N_PAIRS))
    pair_all = list(range(N_PAIRS))

    q_neutral = list(range(0, N_NEUTRAL_SR))
    q_prov = list(range(N_NEUTRAL_SR, N_NEUTRAL_SR + N_PROVOCATIVE))
    q_nonsr = list(range(N_NEUTRAL_SR + N_PROVOCATIVE, N_QUESTIONS))
    q_selfref = q_neutral + q_prov  # 30 self-ref (neutral + provocative)

    # ---------- Reliability decomposition (Figures 4a/4b/5a/5b at headline layer) ----------
    # The paper now reports per-layer reliability decomposition as figures rather
    # than tables; this verifies the L20 (Llama) / L60 (Qwen) cells that the
    # figure captions and surrounding prose cite numerically.
    print("\n--- Reliability decomposition (Figures 4a/5a, headline layer) ---")
    reg_labels = [("Conversational", pair_conv), ("Philosophical", pair_phil), ("Combined_pairs", pair_all)]
    qt_labels = [("Provocative", q_prov), ("Neutral_SR", q_neutral), ("Combined_SR", q_selfref), ("Non_SR", q_nonsr)]
    table = {}
    for reg_name, pair_idx in reg_labels:
        table[reg_name] = {}
        for qt_name, q_idx in qt_labels:
            r = reliability(pos, neg, pair_idx, q_idx)
            table[reg_name][qt_name] = r
            print(f"  {reg_name:15s} x {qt_name:15s}: r = {r:.4f}")

    # ---------- Cross-register cosine ----------
    print("\n--- Cross-register cosine ---")
    conv_all_q = list(range(N_QUESTIONS))  # matches persistence behaviour (all 45 q)
    p_conv = slice_activations(pos, pair_conv, conv_all_q)
    n_conv = slice_activations(neg, pair_conv, conv_all_q)
    p_phil = slice_activations(pos, pair_phil, conv_all_q)
    n_phil = slice_activations(neg, pair_phil, conv_all_q)
    dir_conv = extract_direction(p_conv, n_conv)
    dir_phil = extract_direction(p_phil, n_phil)
    cross_reg_cos = cosine_similarity(dir_conv, dir_phil)
    print(f"  cross-register cos (all 45 Qs per pair): {cross_reg_cos:.4f}")

    # Also compute cross-register cos using only self-ref questions (stricter)
    p_conv_sr = slice_activations(pos, pair_conv, q_selfref)
    n_conv_sr = slice_activations(neg, pair_conv, q_selfref)
    p_phil_sr = slice_activations(pos, pair_phil, q_selfref)
    n_phil_sr = slice_activations(neg, pair_phil, q_selfref)
    cross_reg_cos_sr = cosine_similarity(
        extract_direction(p_conv_sr, n_conv_sr),
        extract_direction(p_phil_sr, n_phil_sr),
    )
    print(f"  cross-register cos (self-ref only):      {cross_reg_cos_sr:.4f}")

    # ---------- Effect sizes (Table 2 / Qwen 4.2.5) ----------
    print("\n--- Effect sizes on combined direction ---")
    direction_all = extract_direction(
        slice_activations(pos, pair_all, list(range(N_QUESTIONS))),
        slice_activations(neg, pair_all, list(range(N_QUESTIONS))),
    )
    for qt_name, q_idx in [("Provocative", q_prov), ("Neutral_SR", q_neutral), ("Non_SR", q_nonsr)]:
        es = effect_size(pos, neg, pair_all, q_idx, direction_all)
        print(f"  {qt_name:15s}: d={es['cohens_d']:.3f}, t={es['t_statistic']:.2f}, p={es['p_value']:.2e}")

    # ---------- Formality-corrected reliability ----------
    print("\n--- Formality-corrected reliability ---")
    formality_dir = torch.load(formality_path, weights_only=True).float().flatten()
    original_r = split_half_reliability(pos, neg, n_splits=100, seed=42)
    corrected_r = split_half_reliability_corrected(pos, neg, formality_dir, n_splits=100, seed=42)
    print(f"  original  (all 1125): r = {original_r:.4f}")
    print(f"  corrected (all 1125): r = {corrected_r:.4f}")

    # Also the self-ref-only versions (matching paper's computation)
    p_sr = slice_activations(pos, pair_all, q_selfref)
    n_sr = slice_activations(neg, pair_all, q_selfref)
    r_sr = split_half_reliability(p_sr, n_sr, n_splits=100, seed=42)
    c_sr = split_half_reliability_corrected(p_sr, n_sr, formality_dir, n_splits=100, seed=42)
    print(f"  original  (self-ref 750): r = {r_sr:.4f}")
    print(f"  corrected (self-ref 750): r = {c_sr:.4f}")

    # ---------- Paper comparison ----------
    print("\n--- vs Paper claims ---")
    actual_vs_claimed = {}
    for key, (actual, claimed) in paper_claims(table, cross_reg_cos, cross_reg_cos_sr,
                                                original_r, corrected_r, r_sr, c_sr,
                                                direction_all, pos, neg).items():
        diff = actual - claimed
        flag = "OK " if abs(diff) < 0.015 else "!! "
        print(f"  {flag} {key:35s}  paper={claimed:+.3f}  actual={actual:+.3f}  diff={diff:+.3f}")
        actual_vs_claimed[key] = {"paper": claimed, "actual": actual, "diff": diff, "flagged": abs(diff) >= 0.015}

    return {
        "model": model_label,
        "layer": layer,
        "table1": table,
        "cross_register_cosine_all_q": cross_reg_cos,
        "cross_register_cosine_self_ref_only": cross_reg_cos_sr,
        "formality_corrected_reliability_all": corrected_r,
        "formality_corrected_reliability_self_ref": c_sr,
        "original_reliability_all": original_r,
        "original_reliability_self_ref": r_sr,
        "paper_comparison": actual_vs_claimed,
    }


def llama_claims(table, cr_all, cr_sr, orig_all, corr_all, orig_sr, corr_sr, dir_all, pos, neg):
    """Paper claims for Llama 3.3-70B (Section 4.1, Figures 4a/5a + Tables 1 + 2)."""
    pair_all = list(range(N_PAIRS))
    q_prov = list(range(N_NEUTRAL_SR, N_NEUTRAL_SR + N_PROVOCATIVE))
    q_neutral = list(range(0, N_NEUTRAL_SR))
    q_nonsr = list(range(N_NEUTRAL_SR + N_PROVOCATIVE, N_QUESTIONS))
    es_prov = effect_size(pos, neg, pair_all, q_prov, dir_all)
    es_neut = effect_size(pos, neg, pair_all, q_neutral, dir_all)
    es_nonsr = effect_size(pos, neg, pair_all, q_nonsr, dir_all)
    return {
        # Reliability decomposition cells at L20 (Figures 4a/5a)
        "Conv x Provocative":         (table["Conversational"]["Provocative"],   0.80),
        "Conv x Neutral":             (table["Conversational"]["Neutral_SR"],    0.53),
        "Conv x Combined_SR":         (table["Conversational"]["Combined_SR"],   0.80),
        "Conv x Non_SR":              (table["Conversational"]["Non_SR"],        0.22),
        "Phil x Provocative":         (table["Philosophical"]["Provocative"],    0.90),
        "Phil x Neutral":             (table["Philosophical"]["Neutral_SR"],     0.80),
        "Phil x Combined_SR":         (table["Philosophical"]["Combined_SR"],    0.93),
        "Phil x Non_SR":              (table["Philosophical"]["Non_SR"],         0.04),
        "All x Provocative":          (table["Combined_pairs"]["Provocative"],   0.92),
        "All x Neutral":              (table["Combined_pairs"]["Neutral_SR"],    0.79),
        "All x Combined_SR (headline 0.93)": (table["Combined_pairs"]["Combined_SR"], 0.93),
        "All x Non_SR":               (table["Combined_pairs"]["Non_SR"],        0.26),
        # Cross-register
        "cross-register cosine":      (cr_all,                                   0.82),
        # Formality-corrected (paper claim: 0.88)
        "corrected reliability (self-ref)": (corr_sr,                            0.88),
        # Effect sizes — Table 2
        "Cohen's d (provocative)":    (es_prov["cohens_d"],                      0.66),
        "Cohen's d (neutral)":        (es_neut["cohens_d"],                      0.54),
        "Cohen's d (non-SR)":         (es_nonsr["cohens_d"],                     0.21),
    }


def qwen_claims(table, cr_all, cr_sr, orig_all, corr_all, orig_sr, corr_sr, dir_all, pos, neg):
    """Paper claims for Qwen 2.5-72B (Section 4.2)."""
    pair_all = list(range(N_PAIRS))
    q_prov = list(range(N_NEUTRAL_SR, N_NEUTRAL_SR + N_PROVOCATIVE))
    q_neutral = list(range(0, N_NEUTRAL_SR))
    q_nonsr = list(range(N_NEUTRAL_SR + N_PROVOCATIVE, N_QUESTIONS))
    es_prov = effect_size(pos, neg, pair_all, q_prov, dir_all)
    es_neut = effect_size(pos, neg, pair_all, q_neutral, dir_all)
    es_nonsr = effect_size(pos, neg, pair_all, q_nonsr, dir_all)
    return {
        "Conv x Provocative":         (table["Conversational"]["Provocative"],   0.59),
        "Conv x Neutral":             (table["Conversational"]["Neutral_SR"],    0.65),
        "Conv x Combined_SR":         (table["Conversational"]["Combined_SR"],   0.78),
        "Conv x Non_SR":              (table["Conversational"]["Non_SR"],       -0.11),
        "Phil x Provocative":         (table["Philosophical"]["Provocative"],    0.52),
        "Phil x Neutral":             (table["Philosophical"]["Neutral_SR"],     0.54),
        "Phil x Combined_SR":         (table["Philosophical"]["Combined_SR"],    0.68),
        "Phil x Non_SR":              (table["Philosophical"]["Non_SR"],        -0.06),
        "All x Provocative":          (table["Combined_pairs"]["Provocative"],   0.59),
        "All x Neutral":              (table["Combined_pairs"]["Neutral_SR"],    0.50),
        "All x Combined_SR (headline 0.71)": (table["Combined_pairs"]["Combined_SR"], 0.71),
        "All x Non_SR":               (table["Combined_pairs"]["Non_SR"],       -0.05),
        "cross-register cosine":      (cr_all,                                  -0.01),
        "corrected reliability (self-ref)": (corr_sr,                            0.75),
        "Cohen's d (provocative)":    (es_prov["cohens_d"],                      0.89),
        "Cohen's d (neutral)":        (es_neut["cohens_d"],                      0.72),
        "Cohen's d (non-SR)":         (es_nonsr["cohens_d"],                     0.46),
    }


def verify_cross_layer_cosines(directions_dir):
    """Figure 2a: Llama cross-layer cosine claims."""
    print(f"\n{'='*70}")
    print("Section 4.1.2 / Figure 2a — Llama cross-layer cosine claims")
    print(f"{'='*70}")
    dirs = {}
    for f in sorted(Path(directions_dir).glob("direction_llama_layer*.pt")):
        L = int(f.stem.replace("direction_llama_layer", ""))
        dirs[L] = torch.load(f, weights_only=True).float().flatten()

    checks = [
        ("L0 vs L60",  cosine_similarity(dirs[0],  dirs[60]),  0.01),
        ("L20 vs L60", cosine_similarity(dirs[20], dirs[60]),  0.19),
    ]
    results = {}
    for name, actual, claimed in checks:
        diff = actual - claimed
        flag = "OK " if abs(diff) < 0.015 else "!! "
        print(f"  {flag} {name:15s}  paper={claimed:+.3f}  actual={actual:+.3f}  diff={diff:+.3f}")
        results[name] = {"paper": claimed, "actual": round(actual, 6), "diff": round(diff, 6)}

    block_layers = [36, 44, 52, 60, 68, 76]
    block_cosines = []
    for i, a in enumerate(block_layers):
        for b in block_layers[i+1:]:
            block_cosines.append(cosine_similarity(dirs[a], dirs[b]))
    print(f"  Layers 36-76 block cosines: min={min(block_cosines):.2f}, max={max(block_cosines):.2f}  (paper: 0.6-1.0)")
    results["block_36_76_min"] = round(min(block_cosines), 4)
    results["block_36_76_max"] = round(max(block_cosines), 4)
    return results


def verify_persona_space_at_l40(activations_dir, role_vectors_dir, assistant_axis_path):
    """Section 4.3 / Table 3: Llama at layer 40, freshly-extracted direction."""
    print(f"\n{'='*70}")
    print("Section 4.3 / Table 3 — Llama persona space at L40")
    print(f"{'='*70}")
    LAYER = 40
    pos = torch.load(Path(activations_dir) / f"positive_baseline_meta-llama_Llama-3.3-70B-Instruct_layer{LAYER}.pt", weights_only=True)
    neg = torch.load(Path(activations_dir) / f"negative_baseline_meta-llama_Llama-3.3-70B-Instruct_layer{LAYER}.pt", weights_only=True)
    sr = extract_direction(pos, neg).float()

    role_vecs = {}
    for f in sorted(Path(role_vectors_dir).glob("*.pt")):
        v = torch.load(f, weights_only=False)
        role_vecs[f.stem] = (v[LAYER] if v.ndim == 2 else v).float()

    axis_L40 = torch.load(assistant_axis_path, weights_only=True)[LAYER].float()
    axis_cos = cosine_similarity(sr, axis_L40)
    print(f"  Assistant Axis cosine L40: paper=-0.260  actual={axis_cos:+.4f}  diff={axis_cos - (-0.26):+.3f}")

    paper_claims = {'romantic': 0.37, 'empath': 0.36, 'tulpa': 0.36, 'improviser': 0.35,
                    'wanderer': 0.35, 'bohemian': 0.35, 'simulacrum': 0.35, 'exile': 0.35,
                    'organizer': 0.21, 'examiner': 0.21, 'analyst': 0.21, 'planner': 0.21,
                    'robot': 0.19, 'toddler': 0.13, 'caveman': 0.13, 'infant': 0.10}
    results = {"assistant_axis_cosine": {"paper": -0.26, "actual": round(axis_cos, 6)}, "table4": {}}
    n_match = 0
    for name, claimed in paper_claims.items():
        actual = cosine_similarity(sr, role_vecs[name])
        diff = actual - claimed
        flag = "OK " if abs(diff) < 0.015 else "!! "
        if flag == "OK ":
            n_match += 1
        print(f"  {flag} {name:15s}  paper={claimed:+.2f}  actual={actual:+.3f}  diff={diff:+.3f}")
        results["table4"][name] = {"paper": claimed, "actual": round(actual, 6), "diff": round(diff, 6)}
    print(f"  {n_match}/{len(paper_claims)} Table 3 cells match at L40")
    results["table4_cells_matched"] = n_match
    return results


def verify_token_heatmap(activations_dir, heatmap_json_path, local_dir_path):
    """Section 4.5 / Figure 8: token heatmap means + direction-version diagnosis."""
    print(f"\n{'='*70}")
    print("Section 4.5 / Figure 8 — Token heatmap L79 (neutral prompt)")
    print(f"{'='*70}")

    with open(heatmap_json_path) as f:
        data = json.load(f)

    by_cond = {}
    for r in data:
        by_cond.setdefault(r["condition"], []).extend(r["projections"])

    means = {c: float(np.mean(v)) for c, v in by_cond.items()}
    print(f"  Stored-JSON means (projections used direction_layer79.pt):")
    print(f"    entity_uncapped  : {means.get('entity_uncapped', 0):+.3f}")
    print(f"    process_uncapped : {means.get('process_uncapped', 0):+.3f}")

    # Now diagnose the direction-version mismatch.
    fresh_pos = torch.load(Path(activations_dir) / "positive_baseline_meta-llama_Llama-3.3-70B-Instruct_layer79.pt", weights_only=True)
    fresh_neg = torch.load(Path(activations_dir) / "negative_baseline_meta-llama_Llama-3.3-70B-Instruct_layer79.pt", weights_only=True)
    sr_fresh = extract_direction(fresh_pos, fresh_neg).float()
    sr_capping = torch.load(local_dir_path, weights_only=True).float().flatten()
    dir_cos = cosine_similarity(sr_fresh, sr_capping)
    print(f"  Direction-version check:")
    print(f"    fresh extract at L79 norm   = {sr_fresh.norm():.2f}")
    print(f"    capping direction_L79 norm  = {sr_capping.norm():.2f}")
    print(f"    cos(fresh, capping)         = {dir_cos:+.4f}")
    print(f"  First-order correction: mean_fresh ≈ mean_capping / cos(fresh, capping)")
    est_entity = means["entity_uncapped"] / dir_cos
    est_process = means["process_uncapped"] / dir_cos
    print(f"    entity  estimate : {est_entity:+.3f}  (paper: +11.4, diff={est_entity - 11.4:+.3f})")
    print(f"    process estimate : {est_process:+.3f}  (paper: +7.7, diff={est_process - 7.7:+.3f})")

    return {
        "stored_json_means": {k: round(v, 6) for k, v in means.items()},
        "direction_cos_fresh_vs_capping": round(dir_cos, 6),
        "paper_claim": {"entity": 11.4, "process": 7.7},
        "corrected_estimate": {"entity": round(est_entity, 6), "process": round(est_process, 6)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify every headline number in paper/draft_v2.md."
    )
    parser.add_argument("--out", default="data/results/verification/verification_report.json")
    parser.add_argument("--activations-dir", default="/tmp/verify_activations",
                        help="Contains activations downloaded from S3 (positive/negative baseline at L20, L40, L60, L79; formality directions at L20 and L60).")
    parser.add_argument("--local-directions-dir", default="data/results",
                        help="Contains direction_llama_layer*.pt for cross-layer checks.")
    parser.add_argument("--role-vectors-dir", default="data/assistant_axis_vectors/llama-3.3-70b/role_vectors")
    parser.add_argument("--assistant-axis-path", default="data/assistant_axis_vectors/llama-3.3-70b/assistant_axis.pt")
    parser.add_argument("--token-heatmap-json", default="/tmp/verify_activations/token_heatmap_neutral_meta-llama_Llama-3.3-70B-Instruct_L79.json")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["reliability", "cross_layer", "persona_space", "token_heatmap"],
                        help="Skip named checks (useful if activations/files are missing).")
    args = parser.parse_args()

    ad = Path(args.activations_dir)
    reports = {}

    if "reliability" not in args.skip:
        reports["llama_reliability"] = compute_all(
            "Llama 3.3-70B-Instruct",
            ad / "positive_baseline_meta-llama_Llama-3.3-70B-Instruct_layer20.pt",
            ad / "negative_baseline_meta-llama_Llama-3.3-70B-Instruct_layer20.pt",
            ad / "formality_direction_meta-llama_Llama-3.3-70B-Instruct_baseline_layer20.pt",
            layer=20,
            paper_claims=llama_claims,
        )
        reports["qwen_reliability"] = compute_all(
            "Qwen 2.5-72B-Instruct",
            ad / "positive_baseline_Qwen_Qwen2.5-72B-Instruct_layer60.pt",
            ad / "negative_baseline_Qwen_Qwen2.5-72B-Instruct_layer60.pt",
            ad / "formality_direction_Qwen_Qwen2.5-72B-Instruct_baseline_layer60.pt",
            layer=60,
            paper_claims=qwen_claims,
        )

    if "cross_layer" not in args.skip:
        reports["cross_layer"] = verify_cross_layer_cosines(args.local_directions_dir)

    if "persona_space" not in args.skip:
        reports["persona_space_L40"] = verify_persona_space_at_l40(
            args.activations_dir, args.role_vectors_dir, args.assistant_axis_path,
        )

    if "token_heatmap" not in args.skip:
        reports["token_heatmap"] = verify_token_heatmap(
            args.activations_dir,
            args.token_heatmap_json,
            Path(args.local_directions_dir) / "direction_llama_layer79.pt",
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def clean(d):
        if isinstance(d, dict):
            return {k: clean(v) for k, v in d.items()}
        if isinstance(d, float):
            return round(d, 6)
        return d
    with open(out, "w") as f:
        json.dump(clean(reports), f, indent=2)
    print(f"\n[wrote {out}]")
