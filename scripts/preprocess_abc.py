import os
import sys
import torch
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.router.topology_check import TopologicalRouter, compute_euler_characteristic

def load_abc_mesh(obj_path):
    mesh = trimesh.load(obj_path, force='mesh', process=False)
    points = mesh.vertices
    normals = mesh.vertex_normals
    return mesh, torch.tensor(points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Precompute OT Mapper indices for ABC Dataset")
    parser.add_argument('--mesh_dir', type=str, default='/media/HDD/mamta_backup/datasets/otno/car-pressure-data/data')
    parser.add_argument('--press_dir', type=str, default='/media/HDD/mamta_backup/datasets/otno/car-pressure-data/data')
    parser.add_argument('--out_dir', type=str, default='/media/HDD/mamta_backup/datasets/otno/car-pressure-data/preprocessed')
    parser.add_argument('--width', type=int, default=64, help="Fixed width for 2D latent grid")
    parser.add_argument('--blur', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    router = TopologicalRouter()

    indices = list(range(1, 800))
    for file_idx in tqdm(indices, desc="Preprocessing ABC Data"):
        mesh_path = os.path.join(args.mesh_dir, f"mesh_{file_idx:03d}.ply")
        press_path = os.path.join(args.press_dir, f"press_{file_idx:03d}.npy")
        
        if not os.path.exists(mesh_path):
             mesh_path = os.path.join(args.mesh_dir, f"mesh_{file_idx}.ply")
             press_path = os.path.join(args.press_dir, f"press_{file_idx}.npy")
             
        if not os.path.exists(mesh_path) or not os.path.exists(press_path):
            continue
            
        out_path = os.path.join(args.out_dir, f"data_{file_idx:03d}.pt")
        if os.path.exists(out_path):
            continue

        raw_mesh, points, normals = load_abc_mesh(mesh_path)
        pressure = torch.from_numpy(np.load(press_path)).float()
        
        num_min = min(len(points), len(pressure))
        points = points[:num_min]
        normals = normals[:num_min]
        pressure = pressure[:num_min]

        # Common OTNO Baseline (Spherical mapping per the paper main method)
        otno_mapper = OT3Dto2DMapper(latent_topology="spherical", width=args.width)
        otno_idx_encoder, otno_idx_decoder, otno_grid_width = otno_mapper.get_otno_indices(points, blur=args.blur)

        # TOPOS Topology-Aware Mapping
        chi = compute_euler_characteristic(mesh=raw_mesh)
        topology = router.route(chi=chi)
        
        if topology == "graph":
            topos_idx_encoder = torch.arange(len(points))
            topos_idx_decoder = torch.arange(len(points))
            topos_grid_width = 0
        else:
            topos_width = args.width
            if topology == "volumetric":
                # Scale 3D voxel width to match 2D node counts roughly (e.g., 64x64 ≈ 16x16x16)
                topos_width = int(round(args.width ** (2/3)))
            
            topos_mapper = OT3Dto2DMapper(latent_topology=topology, width=topos_width)
            topos_idx_encoder, topos_idx_decoder, topos_grid_width = topos_mapper.get_otno_indices(points, blur=args.blur)

        data = {
            'points': points,
            'normals': normals,
            'pressure': pressure,
            'chi': chi,
            'topology': topology,
            'otno_idx_encoder': otno_idx_encoder.cpu(),
            'otno_idx_decoder': otno_idx_decoder.cpu(),
            'otno_grid_width': otno_grid_width,
            'topos_idx_encoder': topos_idx_encoder.cpu(),
            'topos_idx_decoder': topos_idx_decoder.cpu(),
            'topos_grid_width': topos_grid_width
        }
        
        torch.save(data, out_path)

if __name__ == "__main__":
    main()
