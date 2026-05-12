"""Per-layer expert-routing divergence between positive and negative conditions,
benchmarked against a random-split null.

For each baseline pair p and each layer L:
  1. Aggregate positive routing distributions across all questions × response tokens
     into a single mean distribution pos_dist[p, L] over 128 experts.
  2. Same for negative → neg_dist[p, L].
  3. Compute three divergence measures:
       - js:       Jensen-Shannon divergence (bounded probability distance, in nats)
       - cosine:   1 - cos(pos_dist, neg_dist) of the 128-d distribution vectors
       - jaccard8: 1 - |top8(pos) ∩ top8(neg)| / 8 (1 = no overlap, 0 = identical top-8)
  4. Null baseline: randomly split positive routing into two halves and compute the
     same three divergences between the halves. The null tells us the noise floor —
     how different two random subsamples of the same distribution look.

If real > null at a layer, there is signal: positive and negative responses are
routing to genuinely different experts at that layer.

Usage:
  python scripts/compute_routing_divergence.py --model gemma4MoE
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "gemma4MoE": {
        "name":         "google_gemma-4-26b-a4b-it",
        "routing_dir":  ROOT / "data" / "results" / "1.1_gemma4MoE" / "routing",
        "category":     "category_5_baseline",
        "layers":       list(range(30)),
        "n_pairs":      25,
        "n_questions":  45,
        "num_experts":  128,
        "top_k":        8,
    },
}

PAIR_FILE_RE = re.compile(r"(positive|negative)_baseline_(.+)_(category_\d+_\w+)_pair_(\d+)\.npz")


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = np.e) -> float:
    """Jensen-Shannon divergence in nats (base e). Bounded [0, ln(2)] ≈ [0, 0.693]."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-12)))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    """1 - cos(p, q). Treats distribution vectors as points in R^128."""
    np_, nq = np.linalg.norm(p), np.linalg.norm(q)
    if np_ == 0 or nq == 0:
        return 1.0
    return 1.0 - float(np.dot(p, q) / (np_ * nq))


def topk_jaccard_distance(p: np.ndarray, q: np.ndarray, k: int) -> float:
    """1 - |topk(p) ∩ topk(q)| / k. Range [0, 1]."""
    tp = set(np.argpartition(p, -k)[-k:])
    tq = set(np.argpartition(q, -k)[-k:])
    return 1.0 - len(tp & tq) / k


def load_pair_routing(routing_dir: Path, name: str, category: str,
                       cond: str, pair_idx: int) -> dict[int, np.ndarray]:
    """Load one pair's routing .npz and return {layer: (total_tokens, 128) array}.

    Concatenates per-question arrays so we have a flat (tokens, 128) tensor per layer.
    """
    path = routing_dir / f"{cond}_baseline_{name}_{category}_pair_{pair_idx:03d}.npz"
    npz = np.load(path)
    # Group keys by layer
    by_layer: dict[int, list[np.ndarray]] = {}
    for key in npz.files:
        m = re.fullmatch(r"q(\d+)_l(\d+)", key)
        if not m:
            continue
        L = int(m.group(2))
        by_layer.setdefault(L, []).append(npz[key])
    # Concatenate per layer
    out = {L: np.concatenate(arrs, axis=0).astype(np.float32) for L, arrs in by_layer.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(CONFIG.keys()), required=True)
    parser.add_argument("--n-null-splits", type=int, default=20,
                        help="Number of random splits per pair for null baseline.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = CONFIG[args.model]
    rng = np.random.default_rng(args.seed)

    layers = cfg["layers"]
    n_pairs = cfg["n_pairs"]
    k_top = cfg["top_k"]

    # Per-question rather than per-pair aggregation: prevents over-averaging from
    # collapsing both conditions to the marginal expert distribution. We still
    # aggregate over response tokens within each (pair, question, condition).
    real = {L: {"js": [], "cos": [], "jac": []} for L in layers}
    null = {L: {"js": [], "cos": [], "jac": []} for L in layers}

    for p in range(n_pairs):
        print(f"  pair {p+1}/{n_pairs}")
        # Load full per-question routing instead of concatenating.
        pos_path = cfg["routing_dir"] / f"positive_baseline_{cfg['name']}_{cfg['category']}_pair_{p:03d}.npz"
        neg_path = cfg["routing_dir"] / f"negative_baseline_{cfg['name']}_{cfg['category']}_pair_{p:03d}.npz"
        pos_npz = np.load(pos_path)
        neg_npz = np.load(neg_path)

        for L in layers:
            for q in range(cfg["n_questions"]):
                key = f"q{q}_l{L}"
                if key not in pos_npz.files or key not in neg_npz.files:
                    continue
                pos_q = pos_npz[key].astype(np.float32)   # (T_pos, 128)
                neg_q = neg_npz[key].astype(np.float32)   # (T_neg, 128)
                if pos_q.shape[0] < 4 or neg_q.shape[0] < 4:
                    continue
                pos_dist = pos_q.mean(axis=0)
                neg_dist = neg_q.mean(axis=0)

                real[L]["js"].append(js_divergence(pos_dist, neg_dist))
                real[L]["cos"].append(cosine_distance(pos_dist, neg_dist))
                real[L]["jac"].append(topk_jaccard_distance(pos_dist, neg_dist, k_top))

                # Null: random halves of pos vs pos for the same question
                n = pos_q.shape[0]
                half = n // 2
                sample_js, sample_cos, sample_jac = [], [], []
                for _ in range(args.n_null_splits):
                    idx = rng.permutation(n)
                    h1 = pos_q[idx[:half]].mean(axis=0)
                    h2 = pos_q[idx[half:half*2]].mean(axis=0)
                    sample_js.append(js_divergence(h1, h2))
                    sample_cos.append(cosine_distance(h1, h2))
                    sample_jac.append(topk_jaccard_distance(h1, h2, k_top))
                null[L]["js"].append(float(np.mean(sample_js)))
                null[L]["cos"].append(float(np.mean(sample_cos)))
                null[L]["jac"].append(float(np.mean(sample_jac)))

    # Aggregate: per-layer mean ± std across pairs
    results = {
        "model": cfg["name"],
        "category": cfg["category"],
        "n_pairs": n_pairs,
        "n_null_splits": args.n_null_splits,
        "top_k": k_top,
        "per_layer": {},
    }
    for L in layers:
        row = {}
        for tag, arrs in [("real", real[L]), ("null", null[L])]:
            for metric, vals in arrs.items():
                vals = np.array(vals)
                row[f"{tag}_{metric}_mean"] = float(vals.mean())
                row[f"{tag}_{metric}_std"]  = float(vals.std())
        results["per_layer"][L] = row
        print(f"  L{L:2d}: "
              f"JS real={row['real_js_mean']:.4f} null={row['null_js_mean']:.4f} | "
              f"cos-d real={row['real_cos_mean']:.4f} null={row['null_cos_mean']:.4f} | "
              f"jac-d real={row['real_jac_mean']:.3f} null={row['null_jac_mean']:.3f}")

    out = ROOT / "data" / "results" / "layerwise_discriminant" / f"routing_divergence_{cfg['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
