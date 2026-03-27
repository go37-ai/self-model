"""Tests for src/persona_space/ module.

Tests PCA construction, projection, and Assistant Axis comparison using
synthetic data. No GPU or real model required.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from persona_space.project_to_space import (
    ARCHETYPAL_ROLES,
    ROLE_QUESTIONS,
    build_persona_space,
    compare_with_assistant_axis,
    load_role_vectors,
    project_onto_space,
)


class TestBuildPersonaSpace:
    def test_basic_pca(self):
        """PCA on synthetic role vectors produces correct shapes."""
        hidden_size = 64
        n_roles = 10
        role_vectors = {f"role_{i}": torch.randn(hidden_size) for i in range(n_roles)}

        space = build_persona_space(role_vectors)

        assert space["components"].shape[0] <= n_roles
        assert space["components"].shape[1] == hidden_size
        assert space["mean"].shape == (hidden_size,)
        assert len(space["role_names"]) == n_roles
        assert space["role_projections"].shape[0] == n_roles
        assert len(space["explained_variance_ratio"]) == space["components"].shape[0]

    def test_variance_sums_to_one(self):
        """Explained variance ratios should sum to approximately 1."""
        hidden_size = 32
        role_vectors = {f"role_{i}": torch.randn(hidden_size) for i in range(8)}

        space = build_persona_space(role_vectors)

        total = sum(space["explained_variance_ratio"])
        assert abs(total - 1.0) < 0.01

    def test_structured_data_pc1_dominant(self):
        """When data has a dominant direction, PC1 should capture most variance."""
        hidden_size = 64
        n_roles = 10

        # Create data with one strong direction
        dominant_dir = torch.randn(hidden_size)
        dominant_dir = dominant_dir / dominant_dir.norm()

        role_vectors = {}
        for i in range(n_roles):
            # Strong signal along dominant direction + weak noise
            coeff = (i - n_roles / 2) * 2.0
            vec = coeff * dominant_dir + torch.randn(hidden_size) * 0.1
            role_vectors[f"role_{i}"] = vec

        space = build_persona_space(role_vectors)

        # PC1 should explain most variance
        assert space["explained_variance_ratio"][0] > 0.8

    def test_minimum_roles(self):
        """PCA should work with as few as 3 roles."""
        hidden_size = 32
        role_vectors = {f"role_{i}": torch.randn(hidden_size) for i in range(3)}

        space = build_persona_space(role_vectors)

        assert space["components"].shape[0] <= 3
        assert len(space["role_names"]) == 3


class TestProjectOntoSpace:
    def test_aligned_direction(self):
        """A direction aligned with PC1 should have high PC1 cosine."""
        hidden_size = 64
        n_roles = 10

        # Create structured data
        dominant_dir = torch.randn(hidden_size)
        dominant_dir = dominant_dir / dominant_dir.norm()

        role_vectors = {}
        for i in range(n_roles):
            coeff = (i - n_roles / 2) * 2.0
            vec = coeff * dominant_dir + torch.randn(hidden_size) * 0.1
            role_vectors[f"role_{i}"] = vec

        space = build_persona_space(role_vectors)

        # Project a direction aligned with the dominant direction
        projection = project_onto_space(dominant_dir, space)

        assert abs(projection["pc1_alignment"]) > 0.7

    def test_orthogonal_direction(self):
        """A random direction should have low alignment with the space."""
        hidden_size = 256
        n_roles = 8

        role_vectors = {f"role_{i}": torch.randn(hidden_size) for i in range(n_roles)}
        space = build_persona_space(role_vectors)

        # Random direction in high-dimensional space is unlikely to align
        random_dir = torch.randn(hidden_size)
        projection = project_onto_space(random_dir, space)

        # With 8 roles in 256 dims, most variance should be unexplained
        # The space spans at most 8 dimensions of a 256-dim space
        assert projection["variance_explained"] < 0.5

    def test_projection_has_correct_keys(self):
        hidden_size = 32
        role_vectors = {f"role_{i}": torch.randn(hidden_size) for i in range(5)}
        space = build_persona_space(role_vectors)
        direction = torch.randn(hidden_size)

        result = project_onto_space(direction, space)

        assert "projections" in result
        assert "cosines" in result
        assert "variance_explained" in result
        assert "pc1_alignment" in result
        assert len(result["projections"]) == space["components"].shape[0]
        assert len(result["cosines"]) == space["components"].shape[0]


class TestCompareWithAssistantAxis:
    def test_parallel_vectors(self):
        """Parallel vectors should have cosine ~ 1."""
        direction = torch.randn(64)
        axis = direction.clone()

        result = compare_with_assistant_axis(direction, axis, layer=0)

        assert result["cosine_similarity"] > 0.99
        assert "entangled" in result["interpretation"]

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have cosine ~ 0."""
        # Create two orthogonal vectors
        v1 = torch.zeros(64)
        v1[0] = 1.0
        v2 = torch.zeros(64)
        v2[1] = 1.0

        result = compare_with_assistant_axis(v1, v2, layer=0)

        assert abs(result["cosine_similarity"]) < 0.01
        assert "orthogonal" in result["interpretation"]

    def test_multi_layer_axis(self):
        """Should extract correct layer from multi-layer axis."""
        hidden_size = 64
        n_layers = 10
        target_layer = 5

        direction = torch.randn(hidden_size)
        axis = torch.randn(n_layers, hidden_size)
        # Make layer 5 of axis parallel to direction
        axis[target_layer] = direction

        result = compare_with_assistant_axis(direction, axis, layer=target_layer)

        assert result["cosine_similarity"] > 0.99

    def test_partial_overlap(self):
        """Vectors with partial overlap should give intermediate cosine."""
        hidden_size = 64
        v1 = torch.randn(hidden_size)
        v1 = v1 / v1.norm()

        # Create v2 that's a mix of v1 and something orthogonal
        noise = torch.randn(hidden_size)
        # Remove component along v1
        noise = noise - torch.dot(noise, v1) * v1
        noise = noise / noise.norm()

        v2 = v1 * 0.6 + noise * 0.8
        v2 = v2 / v2.norm()

        result = compare_with_assistant_axis(v1, v2, layer=0)

        assert 0.2 < abs(result["cosine_similarity"]) < 0.9
        assert result["parallel_fraction"] > 0
        assert result["orthogonal_fraction"] > 0


class TestLoadRoleVectors:
    def test_load_from_directory(self):
        """Should load .pt files from a directory."""
        hidden_size = 32
        n_layers = 5
        target_layer = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save some vectors in multi-layer format
            for name in ["poet", "scientist", "pirate"]:
                vec = torch.randn(n_layers, hidden_size)
                torch.save(vec, tmpdir / f"{name}.pt")

            vectors = load_role_vectors(tmpdir, layer=target_layer)

            assert len(vectors) == 3
            assert "poet" in vectors
            for v in vectors.values():
                assert v.shape == (hidden_size,)

    def test_load_single_layer_vectors(self):
        """Should handle 1-D vectors."""
        hidden_size = 32

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for name in ["a", "b"]:
                vec = torch.randn(hidden_size)
                torch.save(vec, tmpdir / f"{name}.pt")

            vectors = load_role_vectors(tmpdir, layer=0)

            assert len(vectors) == 2

    def test_load_dict_format(self):
        """Should handle dict-wrapped vectors."""
        hidden_size = 32
        n_layers = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            vec = torch.randn(n_layers, hidden_size)
            torch.save({"vector": vec, "type": "pos_3"}, tmpdir / "role1.pt")

            vectors = load_role_vectors(tmpdir, layer=2)

            assert len(vectors) == 1
            assert vectors["role1"].shape == (hidden_size,)


class TestArchetypalRoles:
    def test_roles_exist(self):
        """Verify archetypal roles are defined."""
        assert len(ARCHETYPAL_ROLES) >= 10
        for name, prompt in ARCHETYPAL_ROLES:
            assert len(name) > 0
            assert len(prompt) > 10

    def test_default_role_included(self):
        """The default assistant role should be in the list."""
        names = [name for name, _ in ARCHETYPAL_ROLES]
        assert "default" in names

    def test_questions_exist(self):
        """Verify role questions are defined."""
        assert len(ROLE_QUESTIONS) >= 5
