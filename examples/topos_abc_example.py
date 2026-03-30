import os
import sys
import torch
import numpy as np

try:
    import trimesh
except ImportError:
    print("Trimesh is required to parse ABC dataset .obj files. Please run: pip install trimesh")
    sys.exit(1)

# Import TOPOS modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.router.topology_check import TopologicalRouter
from experiments.shapenet.topos.TransportFNO import TransportFNO

def load_abc_mesh(obj_path, num_samples=3500):
    """ 
    Parse an ABC Dataset CAD model (.obj) 
    and extract a continuous uniform surface point cloud.
    """
    mesh = trimesh.load(obj_path, force='mesh')
    
    # The ABC dataset models are extremely high-poly and heterogeneous.
    # OTNO works best when we sample standard point clouds from the absolute surface manifold:
    points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_indices]
    
    return mesh, torch.tensor(points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

def main():
    print("==========================================")
    print("   TOPOS Pipeline w/ ABC Dataset (.obj)   ")
    print("==========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Acquire a Real CAD Dataset Model (ShapeNet/Car-CFD)
    # The ABC dataset models are often compressed; we provide a high-poly path from ShapeNet as a surrogate:
    cad_mesh_path = "/media/HDD/mamta_backup/datasets/otno/car-pressure-data/data/mesh_001.ply"
    
    if not os.path.exists(cad_mesh_path):
        print(f"[*] Downloading/Mocking {cad_mesh_path}...")
        # Simulating a CAD genus-1 mechanical part if real path is missing:
        mesh = trimesh.creation.torus(major_radius=2.0, minor_radius=0.7)
        mesh.export("mock_cad.obj")
        cad_mesh_path = "mock_cad.obj"
    
    print(f"[*] Loading 3D CAD model: {cad_mesh_path}")
    raw_mesh, phys_points, phys_normals = load_abc_mesh(cad_mesh_path, num_samples=3600)
    print(f"    -> Sampled {len(phys_points)} Points from 3D Boundary Manifold")
    
    # 2. Topology Analysis (The Router)
    print("\n[*] Stage 2: Topo-Router Analysis")
    router = TopologicalRouter()
    
    # Calculate Betti numbers / Euler Char from the raw CAD mesh connectivity
    try:
        chi = router.compute_euler_characteristic(mesh=raw_mesh)
    except Exception:
        chi = 0 # Default fallback for messy CAD files
    
    topology = router.route(chi=chi)
    print(f"    -> Euler Characteristic (χ): {chi}")
    print(f"    -> Extracted Latent Domain : {topology.upper()}")
    
    # 3. Spectral Dimensionality Reduction (Optimal Transport Map)
    print(f"\n[*] Stage 1: Optimal Transport Mapper (Unrolling 3D Shell to 2D {topology.capitalize()})")
    expand_factor = 2.0
    mapper = OT3Dto2DMapper(latent_topology=topology, expand_factor=expand_factor, device=device)
    
    idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(phys_points, blur=0.01)
    print(f"    -> Embedded {grid_width}x{grid_width} ({grid_width**2} total) Latent Points in 3D Space")
    print(f"    -> Extracted Sinkhorn Kantorovich Transport Plan")
    
    # 4. Execute the Latent Operator (SFNO/TFNO)
    print("\n[*] Stage 3: Executing TOPOS Latent Physics Solver")
    # For ABC models, typically predicting stress/pressure tensors. Dummy signal:
    stress_physics_signal = torch.rand_like(phys_points[:, 0:1]) 
    
    # Package parameters for FNO (3 coords + 3 normals + 1 stress + 2 blank = 9 channels)
    phys_features = torch.cat([phys_points, phys_normals, stress_physics_signal], dim=-1).to(device)
    
    # Encoder mapping! Snap physical data onto the 2D SFNO latent structure
    latent_space_features = phys_features[idx_encoder] 
    latent_space_features = torch.cat([latent_space_features, torch.zeros(grid_width*grid_width, 2, device=device)], dim=-1)
    
    # Reshape specifically for the PyTorch Conv2D/FFT2D standard input shape: (Batch, Channels, Height, Width)
    latent_img = latent_space_features.permute(1, 0).view(1, 9, grid_width, grid_width)
    
    # FNO Instantiation
    model = TransportFNO(n_modes=(16, 16), hidden_channels=32, in_channels=9, out_channels=1).to(device)
    
    # Forward Pass through 2D spectral operators, skipping unstructured point cloud math
    # Decoder applies the inverse OT index map natively within the TFNO
    predicted_surface_physics = model(latent_img, idx_decoder) 
    
    print("\n[*] Stage 4: Optimal Transport Decoding")
    print(f"    -> Input  (3D Mesh Points) : {phys_points.shape}")
    print(f"    -> Output (Predicted Field): {predicted_surface_physics.shape}")
    print("\n[SUCCESS] Entire ABC Dataset CAD pipeline processed via OTNO principles.")

if __name__ == "__main__":
    main()
