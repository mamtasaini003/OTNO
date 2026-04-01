import torch
import numpy as np

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    import ot

class OT3Dto2DMapper:
    """
    Robust 3D Geometry to 2D Square Latent Grid Optimal Transport Mapper.
    
    This implements the core mapping mechanism from the OTNO paper:
    1. Generates a 2D mathematical square grid.
    2. Embeds the 2D grid into a 3D latent surface (Torus or Sphere).
    3. Solves the Sinkhorn Optimal Transport plan linking the complex 3D physical 
       mesh to the 3D embedded mathematical surface.
    4. Extracts Encoder / Decoder mappings via the "Mean" or "Max" strategy.
    """
    def __init__(self, latent_topology="torus", expand_factor=2.0, width=None, device="cuda"):
        self.latent_topology = latent_topology.lower()
        self.expand_factor = expand_factor
        self.width = width
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.HAS_GEOMLOSS = HAS_GEOMLOSS
        
        if not self.HAS_GEOMLOSS:
            print("[Warning] 'geomloss' package not installed. Falling back to POT's CPU Sinkhorn (Slow for large meshes).")
            print("To enable fast GPU mapping: pip install geomloss")

    def _generate_latent_torus(self, n_points, R=1.5, r=1.0):
        """ Embeds a 2D Square grid (width x width) into a 3D Torus. """
        width = self.width if self.width is not None else int(np.sqrt(self.expand_factor * n_points))
        # Create 2D Square grid parameters
        theta = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
        phi = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')

        # Torus Parametrization (embedding into 3D)
        x = (R + r * torch.cos(theta)) * torch.cos(phi)
        y = (R + r * torch.cos(theta)) * torch.sin(phi)
        z = r * torch.sin(theta)

        latent_points_3d = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        return latent_points_3d.to(self.device), width

    def _generate_latent_sphere(self, n_points, R=1.0):
        """ Embeds a 2D Square grid (width x width) into a 3D Sphere. """
        width = self.width if self.width is not None else int(np.sqrt(self.expand_factor * n_points))
        u = torch.linspace(0, 2 * np.pi, width)
        v = torch.linspace(0, np.pi, width)
        u, v = torch.meshgrid(u, v, indexing='ij')

        x = R * torch.cos(u) * torch.sin(v)
        y = R * torch.sin(u) * torch.sin(v)
        z = R * torch.cos(v)

        latent_points_3d = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        return latent_points_3d.to(self.device), width

    def _generate_latent_volume(self, n_points, side=2.0):
        """ Embeds a 3D Cartesian grid (width x width x width) into a 3D Volume. """
        width = self.width if self.width is not None else int(np.cbrt(self.expand_factor * n_points))
        x = torch.linspace(-side/2, side/2, width)
        y = torch.linspace(-side/2, side/2, width)
        z = torch.linspace(-side/2, side/2, width)
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')

        latent_points_3d = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
        return latent_points_3d.to(self.device), width

    def compute_sinkhorn_map(self, physical_mesh_3d, blur=0.01):
        """
        Computes the Optimal Transport Plan from physical to latent.
        
        Parameters:
        - physical_mesh_3d: Tensor of shape (N, 3) representing the physical point cloud.
        """
        physical_mesh_3d = physical_mesh_3d.to(self.device)
        N = physical_mesh_3d.size(0)
        
        if self.latent_topology in ["torus", "toroidal"]:
            latent_mesh_3d, width = self._generate_latent_torus(N)
        elif self.latent_topology in ["sphere", "spherical"]:
            latent_mesh_3d, width = self._generate_latent_sphere(N)
        elif self.latent_topology in ["volumetric"]:
            latent_mesh_3d, width = self._generate_latent_volume(N)
        elif self.latent_topology in ["graph"]:
            # Graph topology does not use Sinkhorn mapping
            return None, None, 0
        else:
            raise ValueError(f"Unknown latent topology: {self.latent_topology}")

        M = latent_mesh_3d.size(0)

        if self.HAS_GEOMLOSS:
            # GEOM LOSS: Fast KeOps GPU Sinkhorn
            # We construct empirical measures
            a = torch.ones(N, device=self.device) / N
            b = torch.ones(M, device=self.device) / M
            
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend="tensorized", potentials=True)
            # geomloss returns potentials when potentials=True
            F, G = loss(a, physical_mesh_3d, b, latent_mesh_3d)
            
            C = torch.cdist(physical_mesh_3d, latent_mesh_3d, p=2) ** 2
            P = torch.exp((F.view(N, 1) + G.view(1, M) - C) / blur) * (a.view(N, 1) * b.view(1, M))
        
        else:
            # POT: CPU Sinkhorn
            a = np.ones(N) / N
            b = np.ones(M) / M
            phys_np = physical_mesh_3d.cpu().numpy()
            lat_np = latent_mesh_3d.cpu().numpy()
            
            print(f"  [OT3Dto2DMapper] Computing POT Sinkhorn (N={N}, M={M})...")
            M_cost = torch.cdist(physical_mesh_3d.cpu(), latent_mesh_3d.cpu(), p=2).numpy() ** 2
            P_numpy = ot.sinkhorn(a, b, M_cost, blur, numItermax=2000)
            P = torch.tensor(P_numpy, device=self.device, dtype=torch.float32)

        return P, latent_mesh_3d, width

    def get_otno_indices(self, physical_mesh_3d, strategy="max", blur=0.01):
        """
        Executes the mapping and extracts the Neural Operator Encoder/Decoder indices.
        Returns:
            idx_encoder: Maps Latent -> Physical 
            idx_decoder: Maps Physical -> Latent
            latent_grid_width: integer representing the N x N 2D latent grid sizing.
        """
        P, latent_mesh_3d, width = self.compute_sinkhorn_map(physical_mesh_3d, blur=blur)

        if strategy == "max":
            # Encoder: For each latent point, find the physical point that maximizes P.
            idx_encoder = torch.argmax(P, dim=0) # shape: (M,)
            # Decoder: For each physical point, find the latent point that maximizes P.
            idx_decoder = torch.argmax(P, dim=1) # shape: (N,)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented yet.")
        
        return idx_encoder, idx_decoder, width


if __name__ == "__main__":
    print("--- 3D Surface to 2D Square Latent Sinkhorn Mapper ---")
    mapper = OT3Dto2DMapper(latent_topology="torus", expand_factor=2.0)
    
    # Generate dummy 3D car mesh (ellipsoid)
    N_phys = 1000
    u = np.random.uniform(0, 2*np.pi, N_phys)
    v = np.random.uniform(0, np.pi, N_phys)
    x = 3.0 * np.cos(u) * np.sin(v)  # length
    y = 1.0 * np.sin(u) * np.sin(v)  # width
    z = 0.5 * np.cos(v)              # height
    dummy_car_3d = torch.tensor(np.stack((x, y, z), axis=-1), dtype=torch.float32)
    
    print(f"Physical 3D Surface Points: {dummy_car_3d.shape[0]}")
    
    idx_enc, idx_dec, latent_width = mapper.get_otno_indices(dummy_car_3d, blur=0.01)
    
    print(f"Generated 2D Square Latent Grid: {latent_width}x{latent_width} ({latent_width**2} total latent points)")
    print(f"Encoder Indices (Latent->Phys) shape: {idx_enc.shape}")
    print(f"Decoder Indices (Phys->Latent) shape: {idx_dec.shape}")
    print("Success! Ready to route 3D meshes to 2D Latent FNOs.")
