import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
import trimesh
import random
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models.fno_spherical import ToroidalTransportFNO
from topos.utils import LpLoss, UnitGaussianNormalizer
from topos.data.ot_mapper_3d import OT3Dto2DMapper

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_torus_normals(width, R=1.5, r=1.0):
    """Compute analytical normals on the torus surface."""
    theta = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
    phi   = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta =  r * torch.cos(theta)

    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi =  (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi =  torch.zeros_like(dx_dphi)

    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi

    normals = torch.stack((nx, ny, nz), dim=-1)
    norm = torch.linalg.norm(normals, dim=2, keepdim=True)
    return normals / norm  # (W, W, 3)


class SyntheticGeometryDatasetOTNO(Dataset):
    """
    Dynamically generates the training dataset entirely from scratch by:
    1. Using Latent Geometry Generators to create synthetic 3D "physical" geometries.
    2. Deforming them sequentially to simulate real variations.
    3. Mapping them back to the pure latent topological grid via OT3Dto2DMapper.
    """
    def __init__(self, cache_dir=None, n_train=500, n_test=111, split='train', expand_factor=2.0, num_points=3586):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.expand_factor = expand_factor
        self.num_points = num_points
        self.mapper = OT3Dto2DMapper(latent_topology="toroidal", expand_factor=expand_factor, width=84)
        
        self.num_samples = n_train if split == 'train' else n_test
        self.split = split
        print(f"[*] SyntheticGeometryDataset (Online OTNO) [{split}] initialized with {self.num_samples} generated samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == 'train' else idx + 1000  # distinct IDs for test
        
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_otno_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)
                
        # --- 1. GENERATE PHYSICAL DATA VIA LATENT GENERATORS ---
        # Generate base torus
        clean_points, _ = self.mapper._generate_latent_torus(self.num_points)
        
        # Deform the generation to create a "physical mesh structure"
        noise = torch.randn_like(clean_points) * 0.05
        scale_x = torch.rand(1, device=clean_points.device) * 0.4 + 0.8 # [0.8, 1.2]
        scale_y = torch.rand(1, device=clean_points.device) * 0.4 + 0.8
        
        points = clean_points.clone()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        points += noise
        points = points.float()
        
        # Approximate normals (vector pointing outwards roughly for a torus-like blob)
        normals = points / (torch.linalg.norm(points, dim=-1, keepdim=True) + 1e-6)
        
        # Generate a synthetic pressure field target (e.g. pressure wave)
        pressure = torch.sin(3 * points[:, 0]) * torch.cos(3 * points[:, 1]) + points[:, 2]

        # --- 2. MAP TO LATENT 2D GEOMETRY ---
        idx_encoder, idx_decoder, grid_width = self.mapper.get_otno_indices(points, blur=0.01)
        
        # --- 3. EXPLICIT LATENT 2D STRUCTURAL DOMAIN ---
        latent_coords, _ = self.mapper._generate_latent_torus(self.num_points)
        latent_coords = latent_coords.view(grid_width, grid_width, 3).float()
        latent_normals = compute_torus_normals(grid_width).float()

        data_dict = {
            'points': points.cpu(),
            'normals': normals.cpu(),
            'pressure': pressure.cpu(),
            'idx_encoder': idx_encoder.long().cpu(),
            'idx_decoder': idx_decoder.long().cpu(),
            'latent_coords': latent_coords.cpu(),
            'latent_normals': latent_normals.cpu(),
            'grid_width': grid_width
        }
        
        if self.cache_dir:
            torch.save(data_dict, cache_path)
            
        return data_dict


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/otno.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    set_seed(config['training'].get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cache_dir = config['dataset'].get('cache_dir', None)
    if cache_dir:
        # Move synthetic subset separately
        cache_dir = os.path.join(cache_dir, 'synthetic')

    train_ds = SyntheticGeometryDatasetOTNO(cache_dir=cache_dir,
                                     n_train=config['dataset']['train_samples'], 
                                     n_test=config['dataset']['test_samples'],
                                     split='train', 
                                     expand_factor=config['dataset']['expand_factor'])
                                     
    test_ds = SyntheticGeometryDatasetOTNO(cache_dir=cache_dir,
                                    n_train=config['dataset']['train_samples'], 
                                    n_test=config['dataset']['test_samples'],
                                    split='test', 
                                    expand_factor=config['dataset']['expand_factor'])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    # ---------------------------------------------------------
    # OTNO Dynamic Online Normalization
    # ---------------------------------------------------------
    print("[*] Estimating normalization statistics over dynamic dataset (up to 50 samples)...")
    pressures_list = []
    transports_list = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        
        points = batch['points'][0]
        normals = batch['normals'][0]
        pressure = batch['pressure'][0]
        idx_encoder = batch['idx_encoder'][0]
        grid_width = batch['grid_width'][0].item()
        latent_coords = batch['latent_coords'][0]
        latent_normals = batch['latent_normals'][0]

        # Construct 9-channel feature Map
        mapped_phys_points = points[idx_encoder].view(grid_width, grid_width, 3)
        mapped_phys_normals = normals[idx_encoder].view(grid_width, grid_width, 3)
        normal_cross = torch.cross(mapped_phys_normals, latent_normals, dim=-1)
        
        transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(2,0,1).unsqueeze(0)
        
        transports_list.append(transports)
        pressures_list.append(pressure)

    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures_list, dim=0), reduce_dim=[0])
    transport_norm = UnitGaussianNormalizer(torch.cat(transports_list, dim=0), reduce_dim=[0, 2, 3])
    pressure_norm.to(device)
    transport_norm.to(device)
    print(f"[*] Normalizers initialized. Pressure mean={pressure_norm.mean.item():.4f}, std={pressure_norm.std.item():.4f}")

    # ---------------------------------------------------------
    # Model Setup
    # ---------------------------------------------------------
    model = ToroidalTransportFNO(
        n_modes=tuple(config['model']['n_modes']),
        hidden_channels=config['model']['hidden_channels'],
        in_channels=9,
        out_channels=config['model']['out_channels'],
        n_layers=config['model']['n_layers'],
        use_mlp=config['model']['use_mlp'],
        mlp=config['model']['mlp'],
        norm=config['model'].get('norm', None),
        domain_padding=config['model'].get('domain_padding', None),
        factorization=config['model'].get('factorization', 'tucker'),
        rank=config['model'].get('rank', 1.0),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    myloss = LpLoss(size_average=False)

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    epochs = config['training']['n_epochs']
    print(f"[*] Training OTNO Online for {epochs} epochs...")
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_l2 = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            points = batch['points'][0].to(device)
            normals = batch['normals'][0].to(device)
            pressure = batch['pressure'][0].to(device)
            idx_encoder = batch['idx_encoder'][0].to(device)
            idx_decoder = batch['idx_decoder'][0].to(device)
            grid_width = batch['grid_width'][0].item()
            latent_coords = batch['latent_coords'][0].to(device)
            latent_normals = batch['latent_normals'][0].to(device)

            # Construct 9-channel feature dynamically
            mapped_phys_points = points[idx_encoder].view(grid_width, grid_width, 3)
            mapped_phys_normals = normals[idx_encoder].view(grid_width, grid_width, 3)
            normal_cross = torch.cross(mapped_phys_normals, latent_normals, dim=-1)
            
            transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(2,0,1).unsqueeze(0)
            
            # Normalize
            transports = transport_norm.encode(transports.clone())
            pressure_normed = pressure_norm.encode(pressure.clone().unsqueeze(0)).squeeze(0)

            out = model(transports, idx_decoder.unsqueeze(0))
            loss = myloss(out, pressure_normed.unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                points = batch['points'][0].to(device)
                normals = batch['normals'][0].to(device)
                pressure = batch['pressure'][0].to(device)
                idx_encoder = batch['idx_encoder'][0].to(device)
                idx_decoder = batch['idx_decoder'][0].to(device)
                grid_width = batch['grid_width'][0].item()
                latent_coords = batch['latent_coords'][0].to(device)
                latent_normals = batch['latent_normals'][0].to(device)

                mapped_phys_points = points[idx_encoder].view(grid_width, grid_width, 3)
                mapped_phys_normals = normals[idx_encoder].view(grid_width, grid_width, 3)
                normal_cross = torch.cross(mapped_phys_normals, latent_normals, dim=-1)
                transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(2,0,1).unsqueeze(0)
                
                transports = transport_norm.encode(transports.clone())
                out = model(transports, idx_decoder.unsqueeze(0))
                out = pressure_norm.decode(out.clone())
                
                test_l2 += myloss(out, pressure.unsqueeze(0)).item()

        train_l2 /= len(train_ds)
        test_l2 /= len(test_ds)
        print(f"Epoch {ep+1}/{epochs}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}, Time: {default_timer()-t1:.2f}s")

if __name__ == "__main__":
    main()
