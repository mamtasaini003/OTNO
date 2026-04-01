"""
Topological Router for TOPOS Architecture.

Computes the Euler characteristic of a mesh and routes it to the
appropriate latent workbench (spherical, toroidal, or volumetric).

Euler Characteristic: χ = V − E + F = 2 − 2g
  - g = 0  (χ = 2)  →  Spherical workbench (2D Latitude/Longitude or HEALPix)
  - g = 1  (χ = 0)  →  Toroidal workbench  (2D Periodic)
  - open   (χ = 1)  →  Volumetric / Padded (3D Cartesian or 2D Padded)
  - graph  (χ fails)→  Graph Neural Operator Fallback (Non-manifold / Intersecting grids)
  - else            →  Volumetric workbench (3D Cartesian)
"""

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def compute_euler_characteristic(mesh=None, V=None, E=None, F=None):
    """Compute the Euler characteristic χ = V − E + F.

    Parameters
    ----------
    mesh : trimesh.Trimesh, optional
        A trimesh mesh object. If provided, V/E/F counts are extracted
        automatically.
    V : int, optional
        Number of vertices (used when mesh is None).
    E : int, optional
        Number of edges (used when mesh is None).
    F : int, optional
        Number of faces (used when mesh is None).

    Returns
    -------
    int
        The Euler characteristic χ.
    """
    if mesh is not None:
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh is required to compute Euler characteristic from a mesh. "
                "Install it via: pip install trimesh"
            )
        n_vertices = len(mesh.vertices)
        n_edges = len(mesh.edges_unique)
        n_faces = len(mesh.faces)
        return n_vertices - n_edges + n_faces

    if V is not None and E is not None and F is not None:
        return V - E + F

    raise ValueError(
        "Either a trimesh mesh or explicit V, E, F counts must be provided."
    )


def compute_genus(chi):
    """Compute the genus from the Euler characteristic.

    Parameters
    ----------
    chi : int or float
        Euler characteristic.

    Returns
    -------
    float
        Genus g = (2 − χ) / 2.
    """
    return (2 - chi) / 2.0


class TopologicalRouter:
    """Routes input meshes to topology-compatible latent workbenches.

    The router determines the genus of the input geometry via its Euler
    characteristic and selects the appropriate FNO branch:
      - genus 0  (χ ≈ 2)  → "spherical"  (Lat/Lon or HEALPix 2D)
      - genus 1  (χ ≈ 0)  → "toroidal"   (2D Periodic)
      - open     (χ ≈ 1)  → "volumetric" (Route to Volumetric or padded Toroidal)
      - non-manifold      → "graph"      (Fallback to GNO message-passing)
      - otherwise         → "volumetric" (3D Cartesian)

    Parameters
    ----------
    chi_tol : float
        Tolerance for classifying Euler characteristic values.
        Default is 0.5.
    """

    SPHERICAL = "spherical"
    TOROIDAL = "toroidal"
    VOLUMETRIC = "volumetric"
    GRAPH = "graph"

    def __init__(self, chi_tol=0.5, require_watertight=False):
        self.chi_tol = chi_tol
        self.require_watertight = require_watertight

    def route(self, mesh=None, chi=None, V=None, E=None, F=None):
        """Determine the appropriate workbench for the given geometry.

        Parameters
        ----------
        mesh : trimesh.Trimesh, optional
            Mesh object (takes precedence over chi/V/E/F).
        chi : int or float, optional
            Pre-computed Euler characteristic.
        V, E, F : int, optional
            Vertex, edge, face counts for manual computation.

        Returns
        -------
        str
            One of "spherical", "toroidal", or "volumetric".
        """
        if mesh is not None:
            if not getattr(mesh, 'is_watertight', True) and self.require_watertight:
                # Disconnected patches, infinite Betti numbers, or tearing
                return self.GRAPH
                
            if chi is None:
                try:
                    chi = compute_euler_characteristic(mesh=mesh)
                except Exception:
                    # Mathematical topological algorithms crashed (often caused by extreme non-manifold edges)
                    return self.GRAPH
        elif chi is None:
            chi = compute_euler_characteristic(V=V, E=E, F=F)
            
        if chi is None:
            # Absolute fallback
            return self.GRAPH
            
        genus = compute_genus(chi)

        if abs(chi - 2) <= self.chi_tol:
            # Genus 0: Closed watertight surface -> SFNO (Spherical)
            return self.SPHERICAL
        elif abs(chi - 0) <= self.chi_tol:
            # Genus 1: Torus-like -> 2D FNO (Toroidal)
            return self.TOROIDAL
        elif abs(chi - 1) <= self.chi_tol:
            # Open mesh (Euler char = 1 for a disk) -> 3D FNO (Volumetric)
            return self.VOLUMETRIC
        elif chi < 0:
            # Higher genus (g > 1): Complex manifold -> 3D FNO (Volumetric)
            # This captures genus 2, 3, etc.
            return self.VOLUMETRIC
        else:
            # Non-standard or extremely high chi (often invalid meshes) -> GNO Fallback
            return self.GRAPH

    def route_batch(self, chi_values):
        """Route a batch of geometries given their Euler characteristics.

        Parameters
        ----------
        chi_values : list or np.ndarray
            Euler characteristics for each sample.

        Returns
        -------
        list of str
            Routing decisions for each sample.
        """
        return [self.route(chi=c) for c in chi_values]

    def __repr__(self):
        return f"TopologicalRouter(chi_tol={self.chi_tol})"
