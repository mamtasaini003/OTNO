"""
TOPOS: Topological Optimal-transport Partitioned Operator Solver.

Unified 4-stage neural operator pipeline:
  Stage 1 — OT Encoder:  Pre-computed diffeomorphic transport T (data module)
  Stage 2 — Router:      Topology-aware workbench selection via Euler char.
  Stage 3 — Solver:      Spectral FNO on the selected latent domain
  Stage 4 — Decoder:     Inverse OT mapping via index lookup + projection

The model supports three branches:
  • "spherical"  — 2D Lat/Lon or HEALPix (genus 0, closed surfaces)
  • "toroidal"   — 2D periodic grid FNO (genus 1, torus-like)
  • "volumetric" — 3D Cartesian grid FNO (bounding box, higher genus, or open meshes)
"""

import torch
import torch.nn as nn

from topos.models.fno_spherical import SphericalTransportFNO, ToroidalTransportFNO
from topos.models.fno_3d_regular import VolumetricFNO
from topos.router.topology_check import TopologicalRouter


class TOPOS(nn.Module):
    """TOPOS: Topological Optimal-transport Partitioned Operator Solver.

    Parameters
    ----------
    spherical_config : dict
        Keyword arguments for SphericalTransportFNO (genus-0 branch).
    volumetric_config : dict or None
        Keyword arguments for VolumetricFNO (volumetric branch).
        If None, the volumetric branch is disabled.
    toroidal_config : dict or None
        Keyword arguments for ToroidalTransportFNO (genus-1 branch).
        If None, the toroidal branch shares the spherical branch.
    chi_tol : float
        Euler characteristic tolerance for routing.
    default_topology : str
        Default topology when auto-detection is not used.
        One of "spherical", "toroidal", "volumetric".
    """

    def __init__(
        self,
        spherical_config,
        volumetric_config=None,
        toroidal_config=None,
        chi_tol=0.5,
        default_topology="spherical",
    ):
        super().__init__()

        # ---- Stage 2: Router ----
        self.router = TopologicalRouter(chi_tol=chi_tol)
        self.default_topology = default_topology

        # ---- Stage 3: Solvers ----
        # Spherical branch (genus 0)
        self.spherical_solver = SphericalTransportFNO(**spherical_config)

        # Toroidal branch (genus 1) — separate weights or shared with spherical
        if toroidal_config is not None:
            self.toroidal_solver = ToroidalTransportFNO(**toroidal_config)
        else:
            self.toroidal_solver = None  # Will fall back to spherical_solver

        # Volumetric branch (3D Cartesian)
        if volumetric_config is not None:
            self.volumetric_solver = VolumetricFNO(**volumetric_config)
        else:
            self.volumetric_solver = None

    def _get_solver(self, topology):
        """Return the appropriate solver for the given topology.

        Parameters
        ----------
        topology : str
            One of "spherical", "toroidal", "volumetric".

        Returns
        -------
        nn.Module
            The FNO solver for this topology branch.
        """
        if topology == "spherical":
            return self.spherical_solver
        elif topology == "toroidal":
            if self.toroidal_solver is not None:
                return self.toroidal_solver
            import warnings
            warnings.warn(
                "[TOPOS WARNING] Toroidal branch requested (χ ≈ 0) but toroidal_config was None. "
                "Falling back to spherical_solver (SFNO). "
                "CAUTION: SFNO uses spherical convolutions instead of standard 2D periodic ones! "
                "Consider initializing TOPOS with a toroidal_config."
            )
            return self.spherical_solver  # Shared architecture
        elif topology == "volumetric":
            if self.volumetric_solver is None:
                raise ValueError(
                    "Volumetric branch requested but not configured. "
                    "Provide volumetric_config in TOPOS constructor."
                )
            return self.volumetric_solver
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def route(self, chi=None, mesh=None, V=None, E=None, F=None):
        """Stage 2: Determine the appropriate workbench.

        Parameters
        ----------
        chi : float or None
            Pre-computed Euler characteristic.
        mesh : trimesh.Trimesh or None
            Mesh for automatic Euler char. computation.
        V, E, F : int or None
            Manual vertex/edge/face counts.

        Returns
        -------
        str
            One of "spherical", "toroidal", "volumetric".
        """
        return self.router.route(mesh=mesh, chi=chi, V=V, E=E, F=F)

    def forward(self, transports, idx_decoder, topology="auto", chi=None):
        """Full TOPOS forward pass.

        Stage 1 (OT Encoder) is pre-computed in the data pipeline.
        Stages 2-4 are executed here.

        Parameters
        ----------
        transports : Tensor
            OT-transported features on the latent grid.
            2D Spherical/Toroidal branch: (B, C, H, W)
            3D Volumetric branch: (B, C, H, W, D)
            Note: If an open mesh (2D grid) is routed to the Volumetric branch,
            it will be dynamically adjusted (zero-padded / unsqueezed).
        idx_decoder : LongTensor
            Shape (n_target,). Maps latent grid indices → mesh vertices.
        topology : str
            "spherical", "toroidal", "volumetric", or "auto".
            If "auto", uses chi to determine routing.
        chi : float or None
            Euler characteristic (required when topology="auto").

        Returns
        -------
        Tensor
            Predicted field on the physical mesh.
        """
        # ---- Stage 2: Route ----
        if topology == "auto":
            if chi is None:
                raise ValueError(
                    "Euler characteristic (chi) must be provided when topology='auto'."
                )
            topology = self.router.route(chi=chi)
        elif chi is not None:
            expected_topology = self.router.route(chi=chi)
            if topology != expected_topology:
                import warnings
                warnings.warn(
                    f"Topology mismatch warning: Forced topology '{topology}' "
                    f"does not match expected topology '{expected_topology}' for Euler characteristic chi={chi}."
                )

        # ---- Stage 3: Solve in latent space ----
        solver = self._get_solver(topology)

        # Ensure spatial dimensionality of transports matches the chosen solver
        expected_dim = len(solver.n_modes)
        actual_dim = transports.dim() - 2  # Subtract batch and channel dimensions
        
        # Add bounds checking warning for the idx_decoder to prevent obscure index runtime errors
        latent_node_count = torch.prod(torch.tensor(transports.shape[2:])).item()
        if idx_decoder.max() >= latent_node_count:
            import warnings
            warnings.warn(
                f"[TOPOS CRITICAL WARNING] Decoder index out of bounds! "
                f"idx_decoder max is {idx_decoder.max().item()} but theoretical latent space "
                f"has only {latent_node_count} nodes for shape {transports.shape}. "
                f"This will cause an IndexError during the inverse OT Stage 4!"
            )

        if expected_dim == 3 and actual_dim == 2:
            import warnings
            warnings.warn(
                f"[TOPOS WARNING] Dynamically unsqueezing 2D latent grid to 3D for Volumetric solver. "
                f"Original shape: {list(transports.shape)}. New shape: {list(transports.shape) + [1]}. "
                f"Verify this 2D->3D zero-padding configuration is intended."
            )
            transports = transports.unsqueeze(-1)
        elif expected_dim == 2 and actual_dim == 3:
            if transports.shape[-1] == 1:
                import warnings
                warnings.warn(
                    f"[TOPOS WARNING] Dynamically squeezing dummy 3D axis (depth=1) for 2D solver. "
                    f"Original shape: {list(transports.shape)}."
                )
                transports = transports.squeeze(-1)
            else:
                raise ValueError(f"[TOPOS ERROR] Cannot trivially cast true 3D volume (depth={transports.shape[-1]}) to 2D solver.")
        elif expected_dim != actual_dim:
            raise ValueError(
                f"[TOPOS ERROR] Dimensionality mismatch for {topology} branch. "
                f"Expected {expected_dim}D spatial input, got {actual_dim}D spatial input. "
                f"(transports shape: {transports.shape})"
            )

        out = solver(transports, idx_decoder)

        # ---- Stage 4: Decode ----
        # The OT decode (inverse transport via index lookup) is already
        # integrated in each solver's forward() method. The output here
        # is the physical-space solution.
        return out

    def __repr__(self):
        branches = ["spherical"]
        if self.toroidal_solver is not None:
            branches.append("toroidal (separate)")
        else:
            branches.append("toroidal (shared)")
        if self.volumetric_solver is not None:
            branches.append("volumetric")
        return (
            f"TOPOS(\n"
            f"  branches={branches},\n"
            f"  router={self.router},\n"
            f"  default_topology='{self.default_topology}'\n"
            f")"
        )
