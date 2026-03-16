"""
Tests for models/topos.py — TOPOS pipeline integration tests.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models.topos import TOPOS


# ============================================================
# Shared configs
# ============================================================

SPHERICAL_CONFIG = dict(
    n_modes=(8, 8),
    hidden_channels=16,
    in_channels=4,
    out_channels=1,
    lifting_channels=32,
    projection_channels=32,
    n_layers=2,
    use_mlp=True,
    mlp={'expansion': 1.0, 'dropout': 0},
    factorization='tucker',
    rank=0.4,
)

VOLUMETRIC_CONFIG = dict(
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


# ============================================================
# Tests
# ============================================================

class TestTOPOSInstantiation:
    def test_spherical_only(self):
        model = TOPOS(spherical_config=SPHERICAL_CONFIG)
        assert model.spherical_solver is not None
        assert model.volumetric_solver is None

    def test_with_volumetric(self):
        model = TOPOS(
            spherical_config=SPHERICAL_CONFIG,
            volumetric_config=VOLUMETRIC_CONFIG,
        )
        assert model.spherical_solver is not None
        assert model.volumetric_solver is not None

    def test_repr(self):
        model = TOPOS(spherical_config=SPHERICAL_CONFIG)
        r = repr(model)
        assert "TOPOS" in r
        assert "spherical" in r


class TestTOPOSForward:
    @pytest.fixture
    def model_full(self):
        return TOPOS(
            spherical_config=SPHERICAL_CONFIG,
            volumetric_config=VOLUMETRIC_CONFIG,
        )

    def test_spherical_forward(self, model_full):
        """2D transport data through spherical branch."""
        Ns = 16
        n_target = 50
        x = torch.randn(1, 4, Ns, Ns)
        idx = torch.randint(0, Ns * Ns, (n_target,))
        out = model_full(x, idx, topology='spherical')
        assert out.numel() == n_target

    def test_toroidal_forward(self, model_full):
        """Toroidal shares spherical solver by default."""
        Ns = 16
        n_target = 50
        x = torch.randn(1, 4, Ns, Ns)
        idx = torch.randint(0, Ns * Ns, (n_target,))
        out = model_full(x, idx, topology='toroidal')
        assert out.numel() == n_target

    def test_volumetric_forward(self, model_full):
        """3D transport data through volumetric branch."""
        Nx, Ny, Nz = 8, 8, 8
        n_target = 50
        x = torch.randn(1, 3, Nx, Ny, Nz)
        idx = torch.randint(0, Nx * Ny * Nz, (n_target,))
        out = model_full(x, idx, topology='volumetric')
        assert out.numel() == n_target


class TestTOPOSRouting:
    @pytest.fixture
    def model_full(self):
        return TOPOS(
            spherical_config=SPHERICAL_CONFIG,
            volumetric_config=VOLUMETRIC_CONFIG,
        )

    def test_route_spherical(self, model_full):
        assert model_full.route(chi=2) == "spherical"

    def test_route_toroidal(self, model_full):
        assert model_full.route(chi=0) == "toroidal"

    def test_route_volumetric(self, model_full):
        assert model_full.route(chi=-2) == "volumetric"

    def test_auto_routing_with_chi(self, model_full):
        """Auto mode should route based on chi."""
        Ns = 16
        n_target = 30
        x = torch.randn(1, 4, Ns, Ns)
        idx = torch.randint(0, Ns * Ns, (n_target,))
        # chi=2 → spherical branch (2D input)
        out = model_full(x, idx, topology='auto', chi=2)
        assert out.numel() == n_target

    def test_auto_requires_chi(self, model_full):
        """Auto mode without chi should raise."""
        x = torch.randn(1, 4, 8, 8)
        idx = torch.randint(0, 64, (10,))
        with pytest.raises(ValueError, match="Euler characteristic"):
            model_full(x, idx, topology='auto')

    def test_volumetric_not_configured_raises(self):
        """Requesting volumetric without configuring it should raise."""
        model = TOPOS(spherical_config=SPHERICAL_CONFIG)
        x = torch.randn(1, 3, 8, 8, 8)
        idx = torch.randint(0, 512, (10,))
        with pytest.raises(ValueError, match="Volumetric branch"):
            model(x, idx, topology='volumetric')
