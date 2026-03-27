"""Tests for src/utils/activation_cache.py.

Tests that don't require a real model use a minimal mock transformer.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from utils.activation_cache import ActivationCache, save_activations, load_activations


class FakeLayer(nn.Module):
    """A minimal 'transformer layer' for testing hooks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return (self.linear(x),)  # Return tuple like real transformer layers


class FakeModel(nn.Module):
    """A minimal model with a list of layers for ActivationCache to find."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(hidden_size) for _ in range(num_layers)]
        )
        self._hidden_size = hidden_size

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


class TestActivationCache:
    def test_find_layers(self):
        model = FakeModel(num_layers=4)
        cache = ActivationCache(model)
        assert len(cache._layer_modules) == 4

    def test_records_activations(self):
        model = FakeModel(num_layers=4, hidden_size=16)
        cache = ActivationCache(model, layers=[0, 2])
        cache.register_hooks()

        # Run a forward pass
        x = torch.randn(1, 5, 16)  # batch=1, seq=5, hidden=16
        with torch.no_grad():
            model(x)

        assert cache.num_recordings == 1
        activations = cache.get_activations(token_position="last")
        assert 0 in activations
        assert 2 in activations
        assert 1 not in activations  # Layer 1 was not requested

        # Shape: (1 recording, 16 hidden_size)
        assert activations[0].shape == (1, 16)

    def test_mean_token_position(self):
        model = FakeModel(num_layers=2, hidden_size=8)
        cache = ActivationCache(model, layers=[0])
        cache.register_hooks()

        x = torch.randn(1, 10, 8)
        with torch.no_grad():
            model(x)

        activations = cache.get_activations(token_position="mean")
        assert activations[0].shape == (1, 8)

    def test_clear(self):
        model = FakeModel(num_layers=2, hidden_size=8)
        cache = ActivationCache(model, layers=[0])
        cache.register_hooks()

        x = torch.randn(1, 5, 8)
        with torch.no_grad():
            model(x)
        assert cache.num_recordings == 1

        cache.clear()
        assert cache.num_recordings == 0

    def test_remove_hooks(self):
        model = FakeModel(num_layers=2, hidden_size=8)
        cache = ActivationCache(model, layers=[0])
        cache.register_hooks()
        assert len(cache._hooks) == 1

        cache.remove_hooks()
        assert len(cache._hooks) == 0

    def test_multiple_recordings(self):
        model = FakeModel(num_layers=2, hidden_size=8)
        cache = ActivationCache(model, layers=[0])
        cache.register_hooks()

        for _ in range(3):
            x = torch.randn(1, 5, 8)
            with torch.no_grad():
                model(x)

        assert cache.num_recordings == 3
        activations = cache.get_activations(token_position="last")
        assert activations[0].shape == (3, 8)

    def test_default_all_layers(self):
        model = FakeModel(num_layers=4)
        cache = ActivationCache(model, layers=None)
        assert len(cache.layers) == 4


class TestSaveLoadActivations:
    def test_round_trip(self):
        activations = {
            0: torch.randn(5, 16),
            2: torch.randn(5, 16),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_activations(activations, Path(tmpdir), prefix="test")
            loaded = load_activations(Path(tmpdir), prefix="test", layers=[0, 2])

            assert set(loaded.keys()) == {0, 2}
            assert torch.allclose(activations[0], loaded[0])
            assert torch.allclose(activations[2], loaded[2])

    def test_load_missing_layers(self):
        activations = {0: torch.randn(5, 16)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_activations(activations, Path(tmpdir), prefix="test")
            loaded = load_activations(Path(tmpdir), prefix="test", layers=[0, 5])

            assert 0 in loaded
            assert 5 not in loaded  # Layer 5 was never saved
