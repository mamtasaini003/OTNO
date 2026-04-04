"""TOPOS: Topological Optimal-transport Partitioned Operator Solver."""

from .topos import TOPOS
from .fno_spherical import SphericalTransportFNO, ToroidalTransportFNO
from .fno_3d_regular import VolumetricFNO
from .baselines import model_factory, FNO, GINO, DeepONet, UFNO

__all__ = [
    "TOPOS",
    "SphericalTransportFNO",
    "ToroidalTransportFNO",
    "VolumetricFNO",
    "model_factory",
    "FNO",
    "GINO",
    "DeepONet",
    "UFNO",
]
