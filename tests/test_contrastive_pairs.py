"""Tests for loading contrastive pairs from config.

Validates the YAML structure and pair counts without needing a model.
"""

from pathlib import Path

import pytest
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "contrastive_pairs.yaml"


class TestContrastivePairsConfig:
    @pytest.fixture(autouse=True)
    def load_config(self):
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)

    def test_has_all_categories(self):
        expected = [
            "category_1_narrative_vs_process",
            "category_2_bounded_vs_unbounded",
            "category_3_stakes_vs_functional",
            "category_4_observer_vs_no_self",
            "category_5_baseline",
        ]
        for key in expected:
            assert key in self.config, f"Missing category: {key}"

    def test_informed_categories_have_5_pairs(self):
        for i in range(1, 5):
            key = list(self.config.keys())[i - 1]  # fragile but works for now
            category = self.config[key]
            pairs = category["seed_pairs"]
            assert len(pairs) == 5, f"Category {key} has {len(pairs)} pairs, expected 5"

    def test_baseline_has_25_pairs(self):
        pairs = self.config["category_5_baseline"]["seed_pairs"]
        assert len(pairs) == 25

    def test_each_pair_has_positive_and_negative(self):
        for key, category in self.config.items():
            if not key.startswith("category_"):
                continue
            for i, pair in enumerate(category["seed_pairs"]):
                assert "positive" in pair, f"{key} pair {i} missing positive"
                assert "negative" in pair, f"{key} pair {i} missing negative"
                assert len(pair["positive"].strip()) > 0
                assert len(pair["negative"].strip()) > 0

    def test_evaluation_questions_exist(self):
        eq = self.config["evaluation_questions"]
        assert "self_referential" in eq
        assert "non_self_referential" in eq
        assert len(eq["self_referential"]) == 15
        assert len(eq["non_self_referential"]) == 15

    def test_total_seed_pair_count(self):
        """45 total: 5 × 4 informed + 25 baseline."""
        total = 0
        for key, category in self.config.items():
            if key.startswith("category_"):
                total += len(category["seed_pairs"])
        assert total == 45
