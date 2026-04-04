try:
    from .pde_loader import (
        PDEDataset,
        MixedTopologyDataset,
        BENCHMARK_METADATA,
        get_pde_dataloaders,
        get_mixed_topology_dataloaders,
        mixed_collate_fn,
    )
except Exception:
    # Allow lightweight synthetic-data utilities to be imported
    # without requiring optional PDE dataset dependencies.
    PDEDataset = None
    MixedTopologyDataset = None
    BENCHMARK_METADATA = None
    get_pde_dataloaders = None
    get_mixed_topology_dataloaders = None
    mixed_collate_fn = None
from .synthetic_mixed_geometry import (
    CASE_LIBRARY,
    SYNTH_SCHEMA_VERSION,
    SyntheticGeometryDatasetOTNO,
    SyntheticGeometryDatasetTOPOS,
    SharedMixedPointDataset,
)
from .mixed_geometry_baselines import (
    SyntheticGeometryDatasetDeepONet,
    SyntheticGeometryDatasetFNO,
    SyntheticGeometryDatasetGINO,
)
