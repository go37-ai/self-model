"""Tests for src/extraction/ modules.

Tests contrastive_pairs loading and extract_vector logic using synthetic data.
No GPU or real model required.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from extraction.contrastive_pairs import (
    ALL_CATEGORIES,
    INFORMED_CATEGORIES,
    NAIVE_CATEGORY,
    get_all_questions,
    get_informed_pairs,
    get_naive_pairs,
    get_pairs_by_category,
    load_evaluation_questions,
    load_seed_pairs,
)
from extraction.extract_vector import extract_all_directions, select_best_layer


class TestContrastivePairsLoading:
    def test_load_all_pairs(self):
        pairs = load_seed_pairs()
        assert len(pairs) == 23  # 5*4 + 3

    def test_load_specific_categories(self):
        pairs = load_seed_pairs(categories=INFORMED_CATEGORIES)
        assert len(pairs) == 20

    def test_pair_structure(self):
        pairs = load_seed_pairs()
        for p in pairs:
            assert "positive" in p
            assert "negative" in p
            assert "category" in p
            assert "label" in p
            assert len(p["positive"]) > 10
            assert len(p["negative"]) > 10

    def test_get_informed_pairs(self):
        all_pairs = load_seed_pairs()
        informed = get_informed_pairs(all_pairs)
        assert len(informed) == 20
        for p in informed:
            assert p["category"] in INFORMED_CATEGORIES

    def test_get_naive_pairs(self):
        all_pairs = load_seed_pairs()
        naive = get_naive_pairs(all_pairs)
        assert len(naive) == 3
        for p in naive:
            assert p["category"] == NAIVE_CATEGORY

    def test_get_pairs_by_category(self):
        pairs = load_seed_pairs()
        by_cat = get_pairs_by_category(pairs)
        assert len(by_cat) == 5
        for cat_key in INFORMED_CATEGORIES:
            assert len(by_cat[cat_key]) == 5
        assert len(by_cat[NAIVE_CATEGORY]) == 3


class TestEvaluationQuestions:
    def test_load_questions(self):
        eq = load_evaluation_questions()
        assert len(eq["self_referential"]) == 15
        assert len(eq["non_self_referential"]) == 15

    def test_get_all_questions(self):
        all_q = get_all_questions()
        assert len(all_q) == 30
        # Self-referential come first
        eq = load_evaluation_questions()
        assert all_q[:15] == eq["self_referential"]
        assert all_q[15:] == eq["non_self_referential"]


class TestExtractAllDirections:
    def test_extracts_at_all_layers(self):
        hidden_size = 32
        n_samples = 10
        layers = [0, 1, 2]

        positive = {l: torch.randn(n_samples, hidden_size) for l in layers}
        negative = {l: torch.randn(n_samples, hidden_size) for l in layers}

        directions = extract_all_directions(positive, negative, layers)
        assert len(directions) == 3
        for l in layers:
            assert directions[l].shape == (hidden_size,)

    def test_skips_missing_layers(self):
        positive = {0: torch.randn(10, 32)}
        negative = {0: torch.randn(10, 32), 1: torch.randn(10, 32)}
        directions = extract_all_directions(positive, negative, [0, 1])
        assert 0 in directions
        assert 1 not in directions  # Missing from positive


class TestSelectBestLayer:
    def test_selects_most_reliable_layer(self):
        hidden_size = 32
        signal = torch.randn(hidden_size)

        # Layer 1 has a strong, reliable signal
        pos_1 = signal.unsqueeze(0).repeat(20, 1) + torch.randn(20, hidden_size) * 0.01
        neg_1 = -signal.unsqueeze(0).repeat(20, 1) + torch.randn(20, hidden_size) * 0.01

        # Layer 0 is pure noise (unreliable)
        pos_0 = torch.randn(20, hidden_size)
        neg_0 = torch.randn(20, hidden_size)

        positive = {0: pos_0, 1: pos_1}
        negative = {0: neg_0, 1: neg_1}

        best_layer, reliabilities = select_best_layer(
            positive, negative, [0, 1], n_splits=50
        )
        assert best_layer == 1
        assert reliabilities[1] > reliabilities[0]

    def test_returns_all_reliabilities(self):
        layers = [0, 1, 2]
        positive = {l: torch.randn(20, 16) for l in layers}
        negative = {l: torch.randn(20, 16) for l in layers}

        _, reliabilities = select_best_layer(positive, negative, layers, n_splits=20)
        assert len(reliabilities) == 3
        for l in layers:
            assert l in reliabilities
