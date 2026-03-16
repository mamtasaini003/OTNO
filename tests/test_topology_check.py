"""
Tests for router/topology_check.py — Euler characteristic and topological routing.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.router.topology_check import compute_euler_characteristic, compute_genus, TopologicalRouter


# ============================================================
# Euler characteristic from V, E, F counts
# ============================================================

class TestEulerCharacteristic:
    def test_sphere_vef(self):
        """A tetrahedron (simplest sphere-like mesh): V=4, E=6, F=4 → χ=2."""
        chi = compute_euler_characteristic(V=4, E=6, F=4)
        assert chi == 2

    def test_cube_vef(self):
        """Cube: V=8, E=12, F=6 → χ=2 (genus 0)."""
        chi = compute_euler_characteristic(V=8, E=12, F=6)
        assert chi == 2

    def test_torus_vef(self):
        """Torus triangulation: V=9, E=27, F=18 → χ=0 (genus 1)."""
        chi = compute_euler_characteristic(V=9, E=27, F=18)
        assert chi == 0

    def test_double_torus_vef(self):
        """Double torus: χ=−2 (genus 2)."""
        chi = compute_euler_characteristic(V=10, E=30, F=18)
        assert chi == -2

    def test_missing_args_raises(self):
        """Must provide either mesh or V/E/F."""
        with pytest.raises(ValueError):
            compute_euler_characteristic()


class TestGenus:
    def test_genus_0(self):
        assert compute_genus(2) == 0.0

    def test_genus_1(self):
        assert compute_genus(0) == 1.0

    def test_genus_2(self):
        assert compute_genus(-2) == 2.0


# ============================================================
# Trimesh-based tests (skip if trimesh not installed)
# ============================================================

trimesh = pytest.importorskip("trimesh")


class TestEulerCharacteristicTrimesh:
    def test_icosphere(self):
        """trimesh icosphere should have χ=2 (genus 0)."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        chi = compute_euler_characteristic(mesh=mesh)
        assert chi == 2

    def test_box(self):
        """trimesh box should have χ=2 (genus 0)."""
        mesh = trimesh.creation.box()
        chi = compute_euler_characteristic(mesh=mesh)
        assert chi == 2


# ============================================================
# Topological Router
# ============================================================

class TestTopologicalRouter:
    def setup_method(self):
        self.router = TopologicalRouter(chi_tol=0.5)

    def test_route_spherical(self):
        assert self.router.route(chi=2) == "spherical"

    def test_route_toroidal(self):
        assert self.router.route(chi=0) == "toroidal"

    def test_route_volumetric(self):
        """χ = −2 (genus 2) → volumetric fallback."""
        assert self.router.route(chi=-2) == "volumetric"

    def test_route_from_vef(self):
        """Route from V/E/F counts directly."""
        assert self.router.route(V=4, E=6, F=4) == "spherical"
        assert self.router.route(V=9, E=27, F=18) == "toroidal"

    def test_route_batch(self):
        results = self.router.route_batch([2, 0, -2, 2])
        assert results == ["spherical", "toroidal", "volumetric", "spherical"]

    def test_route_with_trimesh_icosphere(self):
        mesh = trimesh.creation.icosphere()
        assert self.router.route(mesh=mesh) == "spherical"

    def test_repr(self):
        assert "TopologicalRouter" in repr(self.router)
