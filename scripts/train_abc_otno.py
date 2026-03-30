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
from experiments.shapenet.topos.TransportFNO import TransportFNO
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.utils import LpLoss, UnitGaussianNormalizer

class ABCDataset(Dataset):
    """
    Dataloader for the ABC dataset subset with pressure fields.
    Assumes meshes in .ply and pressures in .npy.
    """
    def __init__(self, mesh_dir, press_dir, num_samples=3600, n_train=500, n_test=111, split='train', expand_factor=2.0):
        self.mesh_dir = mesh_dir
        self.press_dir = press_dir
        self.num_samples = num_samples
        self.expand_factor = expand_factor
        
        # In practice, list files and filter pairs. 
        # For this script we assume mesh_I.ply <-> press_I.npy correspondence.
        # We also need OT maps. Note: We compute these online for fairness if not cached.
        
        # For ABC dataset subset:
        indices = list(range(1, 800)) # Up to 800 samples as seen in ls
        
        # Split
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'test':
            self.indices = indices[n_train:n_train+n_test]
        else:
            self.indices = indices
            
        print(f"[*] ABCDataset [{split}] initialized with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def load_abc_mesh(self, obj_path):
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        points = mesh.vertices
        normals = mesh.vertex_normals
        return mesh, torch.tensor(points, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        mesh_path = os.path.join(self.mesh_dir, f"mesh_{file_idx:03d}.ply")
        press_path = os.path.join(self.press_dir, f"press_{file_idx:03d}.npy")
        
        if not os.path.exists(mesh_path):
             mesh_path = os.path.join(self.mesh_dir, f"mesh_{file_idx}.ply")
             press_path = os.path.join(self.press_dir, f"press_{file_idx}.npy")
             
        # Load mesh & pressure
        raw_mesh, points, normals = self.load_abc_mesh(mesh_path)
        pressure = torch.from_numpy(np.load(press_path)).float()
        
        # Ensure points and pressure match (clip if necessary due to previous preprocessing artifacts)
        num_min = min(len(points), len(pressure))
        points = points[:num_min]
        normals = normals[:num_min]
        pressure = pressure[:num_min]
        
        # OT Mapping (genus-0 for OTNO/TransportFNO baseline)
        mapper = OT3Dto2DMapper(latent_topology="spherical", expand_factor=self.expand_factor)
        idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(points, blur=0.01)
        
        return {
            'points': points,
            'normals': normals,
            'pressure': pressure,
            'idx_encoder': idx_encoder,
            'idx_decoder': idx_decoder,
            'grid_width': grid_width,
            'chi': 2.0 # Constant for spherical baseline
        }

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="OTNO Trainer for ABC Dataset")
    parser.add_argument('--config', type=str, default='configs/abc_comparison.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset
    abc_dir = config['dataset']['path']
    train_ds = ABCDataset(abc_dir, abc_dir, 
                          n_train=config['dataset']['train_samples'], 
                          n_test=config['dataset']['test_samples'],
                          split='train', expand_factor=config['dataset']['expand_factor'])
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 2. Model Initialization (TransportFNO)
    # Using the higher-capacity configs from ot_train.py reference
    model = TransportFNO(
        n_modes=(32, 32),
        hidden_channels=120,
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        n_layers=config['model']['n_layers'],
        use_mlp=config['model']['use_mlp'],
        mlp=config['model']['mlp'],
        norm=config['model']['norm'],
        domain_padding=config['model']['domain_padding'],
        factorization=config['model']['factorization'],
        rank=config['model']['rank']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['training']['learning_rate'], 
                                 weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['training']['step_size'], 
                                                gamma=config['training']['gamma'])
    myloss = LpLoss(d=2, size_average=True)

    # 3. Normalization (Reference from ot_train.py)
    # We estimate stats from the first few batches to avoid a full pass on online mapping
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
        
        cross = torch.cross(points, normals, dim=1)
        phys_features = torch.cat([points, normals, cross], dim=-1)
        latent_features = phys_features[idx_encoder] 
        latent_img = latent_features.permute(1, 0).view(1, 9, grid_width, grid_width)
        
        pressures_list.append(pressure.cpu())
        transports_list.append(latent_img.cpu())
    
    pressure_norm = UnitGaussianNormalizer(torch.stack(pressures_list), reduce_dim=[0, 1])
    transport_norm = UnitGaussianNormalizer(torch.cat(transports_list, dim=0), reduce_dim=[0, 2, 3])
    pressure_norm.to(device)
    transport_norm.to(device)
    print(f"[*] Normalizers initialized. Pressure stats: {pressure_norm.mean.item():.4f}, {pressure_norm.std.item():.4f}")

    print(f"[*] Training OTNO on ABC Dataset...")
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
            
            # Pack input
            cross = torch.cross(points, normals, dim=1)
            phys_features = torch.cat([points, normals, cross], dim=-1)
            
            # Encoder map
            latent_features = phys_features[idx_encoder] 
            latent_img = latent_features.permute(1, 0).view(1, 9, grid_width, grid_width)
            
            # Normalize
            latent_img = transport_norm.encode(latent_img)
            target_pressure = pressure_norm.encode(pressure)
            
            # Model forward
            predict = model(latent_img, idx_decoder)
            
            loss = myloss(predict.view(1, -1), target_pressure.view(1, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")

        scheduler.step()
        t2 = default_timer()
        print(f"Epoch {ep+1}/{config['training']['n_epochs']}, Time: {t2-t1:.2f}s, Avg Loss: {total_loss/len(train_loader):.6f}")

if __name__ == "__main__":
    main()
