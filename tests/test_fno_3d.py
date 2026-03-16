"""
Tests for models/fno_3d_regular.py — VolumetricFNO and grid helpers.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models.fno_3d_regular import VolumetricFNO, create_cartesian_grid


class TestCartesianGrid:
    def test_shape(self):
        grid = create_cartesian_grid(8, 8, 8)
        assert grid.shape == (3, 8, 8, 8)

    def test_range(self):
        grid = create_cartesian_grid(16, 16, 16)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_asymmetric(self):
        grid = create_cartesian_grid(8, 12, 16)
        assert grid.shape == (3, 8, 12, 16)


class TestVolumetricFNO:
    @pytest.fixture
    def model(self):
        return VolumetricFNO(
            n_modes=(4, 4, 4),
            hidden_channels=16,
            in_channels=3,
            out_channels=1,
            lifting_channels=32,
            projection_channels=32,
            n_layers=2,
            use_mlp=True,
            mlp={'expansion': 1.0, 'dropout': 0},
            factorization='tucker',
            rank=0.4,
        )

    def test_instantiation(self, model):
        assert model is not None

    def test_forward_shape(self, model):
        """Forward pass with synthetic 3D input → verify output shape."""
        Nx, Ny, Nz = 8, 8, 8
        n_target = 100
        x = torch.randn(1, 3, Nx, Ny, Nz)
        idx = torch.randint(0, Nx * Ny * Nz, (n_target,))
        out = model(x, idx)
        # Output should have n_target points with out_channels=1
        assert out.numel() == n_target

    def test_parameter_count(self, model):
        """Model should have trainable parameters."""
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
