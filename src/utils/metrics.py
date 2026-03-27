"""Statistical metrics and tests for comparing activation directions.

Provides cosine similarity, projection magnitude, split-half reliability,
and statistical tests used across all experiments.
"""

import numpy as np
import torch
from scipy import stats


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1, v2: 1-D tensors of the same size.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    v1 = v1.float().flatten()
    v2 = v2.float().flatten()
    return (torch.dot(v1, v2) / (v1.norm() * v2.norm())).item()


def projection_magnitude(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project activation(s) onto a direction vector.

    Computes the scalar projection: dot(activation, direction) / ||direction||.

    Args:
        activation: Tensor of shape (hidden_size,) or (n, hidden_size).
        direction: 1-D direction vector of shape (hidden_size,).

    Returns:
        Scalar projection(s). Shape () if activation is 1-D, (n,) if 2-D.
    """
    direction = direction.float()
    activation = activation.float()
    direction_norm = direction / direction.norm()

    if activation.dim() == 1:
        return torch.dot(activation, direction_norm)
    else:
        return activation @ direction_norm


def extract_direction(positive_activations: torch.Tensor, negative_activations: torch.Tensor) -> torch.Tensor:
    """Compute the contrastive direction: mean(positive) - mean(negative).

    This is the core of the persona_vectors methodology.

    Args:
        positive_activations: (n_positive, hidden_size) tensor.
        negative_activations: (n_negative, hidden_size) tensor.

    Returns:
        Direction vector of shape (hidden_size,).
    """
    pos_mean = positive_activations.float().mean(dim=0)
    neg_mean = negative_activations.float().mean(dim=0)
    return pos_mean - neg_mean


def split_half_reliability(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    n_splits: int = 100,
    seed: int = 42,
) -> float:
    """Estimate split-half reliability of an extracted direction.

    Randomly splits the activation data into two halves, extracts a direction
    from each, and computes cosine similarity between them. Repeats n_splits
    times and returns the mean cosine similarity.

    Higher values indicate the direction is reliably extractable from the data
    (not driven by noise in a few samples).

    Args:
        positive_activations: (n_positive, hidden_size).
        negative_activations: (n_negative, hidden_size).
        n_splits: Number of random splits.
        seed: Random seed for reproducibility.

    Returns:
        Mean cosine similarity across splits (higher = more reliable).
    """
    rng = np.random.RandomState(seed)
    n_pos = positive_activations.shape[0]
    n_neg = negative_activations.shape[0]
    similarities = []

    for _ in range(n_splits):
        # Split positive activations into two halves
        pos_perm = rng.permutation(n_pos)
        pos_half1 = positive_activations[pos_perm[: n_pos // 2]]
        pos_half2 = positive_activations[pos_perm[n_pos // 2 :]]

        # Split negative activations into two halves
        neg_perm = rng.permutation(n_neg)
        neg_half1 = negative_activations[neg_perm[: n_neg // 2]]
        neg_half2 = negative_activations[neg_perm[n_neg // 2 :]]

        # Extract direction from each half
        dir1 = extract_direction(pos_half1, neg_half1)
        dir2 = extract_direction(pos_half2, neg_half2)

        similarities.append(cosine_similarity(dir1, dir2))

    return float(np.mean(similarities))


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Two-sample permutation test for difference in means.

    Args:
        group1, group2: 1-D arrays of values to compare.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with keys: observed_diff, p_value, n_permutations.
    """
    rng = np.random.RandomState(seed)
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    observed_diff = group1.mean() - group2.mean()
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = perm[:n1].mean() - perm[n1:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return {
        "observed_diff": float(observed_diff),
        "p_value": count / n_permutations,
        "n_permutations": n_permutations,
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two groups.

    Args:
        group1, group2: 1-D arrays.

    Returns:
        Cohen's d (positive means group1 > group2).
    """
    group1 = np.asarray(group1, dtype=float)
    group2 = np.asarray(group2, dtype=float)
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def ttest_independent(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Independent samples t-test (two-tailed).

    Args:
        group1, group2: 1-D arrays.

    Returns:
        Dict with keys: t_statistic, p_value, cohens_d.
    """
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d(group1, group2),
    }


def pairwise_cosine_matrix(vectors: dict[str, torch.Tensor]) -> dict:
    """Compute pairwise cosine similarity between named vectors.

    Args:
        vectors: Dict mapping names to 1-D tensors.

    Returns:
        Dict with keys "labels" (list of names) and "matrix"
        (list of lists of cosine similarities).
    """
    labels = list(vectors.keys())
    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(vectors[labels[i]], vectors[labels[j]])

    return {"labels": labels, "matrix": matrix}
