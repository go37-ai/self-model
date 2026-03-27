"""Tests for src/utils/metrics.py.

These tests use synthetic data — no model or GPU required.
"""

import numpy as np
import pytest
import torch

from utils.metrics import (
    cohens_d,
    cosine_similarity,
    extract_direction,
    pairwise_cosine_matrix,
    permutation_test,
    projection_magnitude,
    split_half_reliability,
    ttest_independent,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = torch.randn(128)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = torch.randn(128)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_different_dtypes(self):
        v1 = torch.randn(64, dtype=torch.float16)
        v2 = torch.randn(64, dtype=torch.float32)
        # Should not raise — both get cast to float32
        result = cosine_similarity(v1, v2)
        assert -1.0 <= result <= 1.0


class TestProjectionMagnitude:
    def test_aligned_projection(self):
        direction = torch.tensor([1.0, 0.0, 0.0])
        activation = torch.tensor([3.0, 4.0, 0.0])
        proj = projection_magnitude(activation, direction)
        assert proj.item() == pytest.approx(3.0, abs=1e-5)

    def test_batch_projection(self):
        direction = torch.tensor([1.0, 0.0])
        activations = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        proj = projection_magnitude(activations, direction)
        assert proj.shape == (2,)
        assert proj[0].item() == pytest.approx(2.0, abs=1e-5)
        assert proj[1].item() == pytest.approx(4.0, abs=1e-5)


class TestExtractDirection:
    def test_basic_direction(self):
        pos = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        neg = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
        direction = extract_direction(pos, neg)
        # Mean pos = [3, 0], mean neg = [0, 3], diff = [3, -3]
        assert direction[0].item() == pytest.approx(3.0, abs=1e-5)
        assert direction[1].item() == pytest.approx(-3.0, abs=1e-5)

    def test_output_shape(self):
        hidden_size = 256
        pos = torch.randn(10, hidden_size)
        neg = torch.randn(10, hidden_size)
        direction = extract_direction(pos, neg)
        assert direction.shape == (hidden_size,)


class TestSplitHalfReliability:
    def test_perfect_signal(self):
        # With a strong, consistent signal, split-half reliability should be high
        hidden_size = 64
        signal = torch.randn(hidden_size)
        pos = signal.unsqueeze(0).repeat(20, 1) + torch.randn(20, hidden_size) * 0.01
        neg = -signal.unsqueeze(0).repeat(20, 1) + torch.randn(20, hidden_size) * 0.01
        reliability = split_half_reliability(pos, neg, n_splits=50)
        assert reliability > 0.9

    def test_pure_noise(self):
        # With random data, reliability should be low
        pos = torch.randn(20, 64)
        neg = torch.randn(20, 64)
        reliability = split_half_reliability(pos, neg, n_splits=50)
        assert reliability < 0.5

    def test_deterministic_with_seed(self):
        pos = torch.randn(20, 64)
        neg = torch.randn(20, 64)
        r1 = split_half_reliability(pos, neg, seed=123)
        r2 = split_half_reliability(pos, neg, seed=123)
        assert r1 == r2


class TestPermutationTest:
    def test_significant_difference(self):
        rng = np.random.RandomState(42)
        group1 = rng.normal(10, 1, 50)
        group2 = rng.normal(0, 1, 50)
        result = permutation_test(group1, group2, n_permutations=1000, seed=42)
        assert result["p_value"] < 0.01
        assert result["observed_diff"] > 0

    def test_no_difference(self):
        rng = np.random.RandomState(42)
        group1 = rng.normal(0, 1, 50)
        group2 = rng.normal(0, 1, 50)
        result = permutation_test(group1, group2, n_permutations=1000, seed=42)
        assert result["p_value"] > 0.05


class TestCohensD:
    def test_large_effect(self):
        group1 = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        group2 = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
        d = cohens_d(group1, group2)
        assert d > 0.8  # Large effect

    def test_zero_effect(self):
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(group, group)
        assert d == pytest.approx(0.0, abs=1e-10)


class TestTtestIndependent:
    def test_returns_dict(self):
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = ttest_independent(group1, group2)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "cohens_d" in result


class TestPairwiseCosineMatrix:
    def test_diagonal_is_ones(self):
        vectors = {
            "a": torch.randn(64),
            "b": torch.randn(64),
            "c": torch.randn(64),
        }
        result = pairwise_cosine_matrix(vectors)
        assert len(result["labels"]) == 3
        for i in range(3):
            assert result["matrix"][i][i] == pytest.approx(1.0, abs=1e-5)

    def test_symmetric(self):
        vectors = {"x": torch.randn(64), "y": torch.randn(64)}
        result = pairwise_cosine_matrix(vectors)
        assert result["matrix"][0][1] == pytest.approx(result["matrix"][1][0], abs=1e-5)
