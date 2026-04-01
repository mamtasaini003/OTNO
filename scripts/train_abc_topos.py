import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
import trimesh
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset

# Add path for models modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models import TOPOS
# from topos.models.topos import TOPOS
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.router.topology_check import TopologicalRouter, compute_euler_characteristic
from topos.utils import LpLoss, UnitGaussianNormalizer

class ABCDatasetTOPOS(Dataset):
    """
    Dataloader for the ABC dataset subset with pressure fields + Topology Routing.
    Assumes meshes in .ply and pressures in .npy.
    """
    def __init__(self, mesh_dir, press_dir, num_samples=3600, n_train=500, n_test=111, split='train', expand_factor=2.0):
        self.mesh_dir = mesh_dir
        self.press_dir = press_dir
        self.prep_dir = os.path.join(os.path.dirname(mesh_dir), 'preprocessed')
        self.num_samples = num_samples
        self.expand_factor = expand_factor
        self.router = TopologicalRouter()
        
        indices = list(range(1, 800)) 
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'test':
            self.indices = indices[n_train:n_train+n_test]
        else:
            self.indices = indices
            
        print(f"[*] ABCDatasetTOPOS [{split}] initialized with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def load_abc_mesh(self, obj_path):
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        points = mesh.vertices
        normals = mesh.vertex_normals
        return mesh, torch.tensor(points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        
        # 1. Try loading from precomputed cache
        data_path = os.path.join(self.prep_dir, f"data_{file_idx:03d}.pt")
        if os.path.exists(data_path):
            data = torch.load(data_path)
            return {
                'points': data['points'],
                'normals': data['normals'],
                'pressure': data['pressure'],
                'idx_encoder': data['topos_idx_encoder'].long(),
                'idx_decoder': data['topos_idx_decoder'].long(),
                'grid_width': data['topos_grid_width'],
                'topology': data['topology'],
                'chi': data['chi']
            }
        
        # 2. Fallback to online computation if missing
        mesh_path = os.path.join(self.mesh_dir, f"mesh_{file_idx:03d}.ply")
        press_path = os.path.join(self.press_dir, f"press_{file_idx:03d}.npy")
        
        if not os.path.exists(mesh_path):
             mesh_path = os.path.join(self.mesh_dir, f"mesh_{file_idx}.ply")
             press_path = os.path.join(self.press_dir, f"press_{file_idx}.npy")
             
        # Load mesh & pressure
        raw_mesh, points, normals = self.load_abc_mesh(mesh_path)
        pressure = torch.from_numpy(np.load(press_path)).float()
        
        # Ensure points and pressure match
        num_min = min(len(points), len(pressure))
        points = points[:num_min]
        normals = normals[:num_min]
        pressure = pressure[:num_min]
        
        # TOPOS Stage 2: Routing
        chi = compute_euler_characteristic(mesh=raw_mesh)
        topology = self.router.route(chi=chi)
        
        # TOPOS Stage 1: OT Mapping (Topology-aware)
        if topology == "graph":
            # GNO branch uses direct graph operations (no OT map)
            # Return dummy values for dataloader consistency
            idx_encoder = torch.arange(len(points))
            idx_decoder = torch.arange(len(points))
            grid_width = 0
        else:
            # Scale 3D voxel width to match 2D node counts (e.g., 64x64 ≈ 16x16x16)
            scaled_width = 64
            if topology == "volumetric":
                scaled_width = 16 
            
            mapper = OT3Dto2DMapper(latent_topology=topology, expand_factor=self.expand_factor, width=scaled_width)
            idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(points, blur=0.01)
        
        return {
            'points': points,
            'normals': normals,
            'pressure': pressure,
            'idx_encoder': idx_encoder.long(),
            'idx_decoder': idx_decoder.long(),
            'grid_width': grid_width,
            'topology': topology,
            'chi': chi
        }

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TOPOS Unified Trainer for ABC Dataset")
    parser.add_argument('--config', type=str, default='configs/abc_comparison.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset
    abc_dir = config['dataset']['path']
    train_ds = ABCDatasetTOPOS(abc_dir, abc_dir, 
                               n_train=config['dataset']['train_samples'], 
                               n_test=config['dataset']['test_samples'],
                               split='train', expand_factor=config['dataset']['expand_factor'])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 2. Model Initialization (TOPOS with Spherical/Toroidal/Volumetric branches)
    if 'spherical_config' in config['model']:
        # Branch-specific definition format (topos.yaml)
        spherical_config = config['model']['spherical_config']
        volumetric_config = config['model']['volumetric_config']
        toroidal_config = config['model'].get('toroidal_config', None)
        
        # Ensure tuples for n_modes
        spherical_config['n_modes'] = tuple(spherical_config['n_modes'])
        if volumetric_config:
            volumetric_config['n_modes'] = tuple(volumetric_config['n_modes'])
        if toroidal_config:
            toroidal_config['n_modes'] = tuple(toroidal_config['n_modes'])
    else:
        # Flat benchmark format (abc_comparison.yaml)
        model_config = {
            'n_modes': tuple(config['model']['n_modes']),
            'hidden_channels': config['model']['hidden_channels'],
            'in_channels': config['model']['in_channels'],
            'out_channels': config['model']['out_channels'],
            'n_layers': config['model']['n_layers'],
            'use_mlp': config['model']['use_mlp'],
            'mlp': config['model']['mlp'],
            'norm': config['model'].get('norm', None),
            'domain_padding': config['model'].get('domain_padding', None),
            'factorization': config['model'].get('factorization', 'tucker'),
            'rank': config['model'].get('rank', 1.0)
        }
        
        spherical_config = model_config
        toroidal_config = model_config

        # Volumetric branch needs 3D modes.
        volumetric_config = model_config.copy()
        volumetric_config['n_modes'] = (model_config['n_modes'][0]//2, model_config['n_modes'][1]//2, 1)

    # Graph Branch Config (Fallback for non-manifold topologies)
    graph_config = {
        'in_channels': config['model']['in_channels'],
        'out_channels': config['model']['out_channels'],
        'hidden_channels': config['model']['hidden_channels'] // 3  # Lightweight fallback
    }

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=toroidal_config,
        volumetric_config=volumetric_config,
        graph_config=graph_config
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['training']['learning_rate'], 
                                 weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['training']['step_size'], 
                                                gamma=config['training']['gamma'])
    myloss = LpLoss(d=2, size_average=True)

    # 3. Normalization estimation
    print("[*] Estimating normalization statistics...")
    pressures_list = []
    transports_list = []
    for i in range(min(50, len(train_loader))):
        batch = next(iter(train_loader))
        points = batch['points'][0].to(device)
        normals = batch['normals'][0].to(device)
        pressure = batch['pressure'][0].to(device)
        idx_encoder = batch['idx_encoder'][0].to(device)
        grid_width = batch['grid_width'][0].item()
        topology = batch['topology'][0]
        
        cross = torch.cross(points, normals, dim=1)
        phys_features = torch.cat([points, normals, cross], dim=-1)
        
        if grid_width > 0:
            latent_features = phys_features[idx_encoder]
            latent_img = latent_features.permute(1, 0).view(1, 9, grid_width, grid_width)
            transports_list.append(latent_img.cpu())
        
        pressures_list.append(pressure.cpu())
    
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures_list, dim=0), reduce_dim=[0])
    if len(transports_list) > 0:
        transport_norm = UnitGaussianNormalizer(torch.cat(transports_list, dim=0), reduce_dim=[0, 2, 3])
        transport_norm.to(device)
    else:
        transport_norm = None
    pressure_norm.to(device)
    print(f"[*] Normalizers initialized. Pressure stats: {pressure_norm.mean.item():.4f}, {pressure_norm.std.item():.4f}")

    print(f"[*] Training TOPOS on ABC Dataset...")
    for ep in range(config['training']['n_epochs']):
        t1 = default_timer()
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            points = batch['points'][0].to(device)
            normals = batch['normals'][0].to(device)
            pressure = batch['pressure'][0].to(device)
            idx_encoder = batch['idx_encoder'][0].to(device)
            idx_decoder = batch['idx_decoder'][0].to(device)
            grid_width = batch['grid_width'][0].item()
            topology = batch['topology'][0]
            chi = batch['chi'][0].item()
            
            # Pack input features
            cross = torch.cross(points, normals, dim=1)
            phys_features = torch.cat([points, normals, cross], dim=-1)
            
            # Step 1: Mapping
            if topology == "graph":
                latent_img = None
                idx_decoder = idx_decoder.long()
            elif topology == "volumetric":
                # OT Mapping
                latent_features = phys_features[idx_encoder] 
                # Reshape to true 3D Volume (W x W x W)
                latent_img = latent_features.permute(1, 0).view(1, 9, grid_width, grid_width, grid_width)
                if transport_norm is not None:
                    # Normalize securely with 3D broadcasting
                    mean_3d = transport_norm.mean.view(1, 9, 1, 1, 1)
                    std_3d = transport_norm.std.view(1, 9, 1, 1, 1)
                    latent_img = (latent_img - mean_3d) / (std_3d + transport_norm.eps)
            else:
                # OT Mapping
                latent_features = phys_features[idx_encoder] 
                # Reshape to 2D Grid (W x W)
                latent_img = latent_features.permute(1, 0).view(1, 9, grid_width, grid_width)
                if transport_norm is not None:
                    latent_img = transport_norm.encode(latent_img.clone())
                
            target_pressure = pressure_norm.encode(pressure.clone())
            
            # Step 2-4: TOPOS Internal Forward (pass points for graph branch)
            predict = model(transports=latent_img, 
                            idx_decoder=idx_decoder, 
                            points=points.unsqueeze(0),
                            features=phys_features.unsqueeze(0),
                            topology=topology, 
                            chi=chi)
            
            # Decode predictions to compute relative L2 loss on actual physical values
            predict_unnorm = pressure_norm.decode(predict.clone())
            loss = myloss(predict_unnorm.view(1, -1), pressure.view(1, -1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f} | Topology: {topology}")

        scheduler.step()
        t2 = default_timer()
        print(f"Epoch {ep+1}/{config['training']['n_epochs']}, Time: {t2-t1:.2f}s, Avg Loss: {total_loss/len(train_loader):.6f}")

if __name__ == "__main__":
    main()
