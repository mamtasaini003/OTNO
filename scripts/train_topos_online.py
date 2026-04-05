import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
import random
from collections import defaultdict
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models import TOPOS
from topos.utils import LpLoss, UnitGaussianNormalizer, prepare_model_for_devices, resolve_device
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.router.topology_check import TopologicalRouter, compute_euler_characteristic

SYNTH_SCHEMA_VERSION = 2
CASE_LIBRARY = [
    {"name": "spherical", "chi": 2.0, "latent_topology": "spherical"},
    {"name": "toroidal", "chi": 0.0, "latent_topology": "toroidal"},
    {"name": "open_surface", "chi": 1.0, "latent_topology": "volumetric"},
    {"name": "high_genus", "chi": -2.0, "latent_topology": "volumetric"},
]


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

def compute_sphere_normals(latent_coords):
    # Latent coords of a unit sphere are its own normals.
    norm = torch.linalg.norm(latent_coords, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return latent_coords / norm


def sample_or_repeat_points(points, target_n, generator):
    """Sample exactly target_n points from a point cloud."""
    n = points.shape[0]
    if n == target_n:
        return points
    if n > target_n:
        idx = torch.randperm(n, generator=generator, device=points.device)[:target_n]
        return points[idx]
    extra = target_n - n
    idx_extra = torch.randint(0, n, (extra,), generator=generator, device=points.device)
    return torch.cat([points, points[idx_extra]], dim=0)


def synthetic_pressure(points, topology):
    """Topology-conditioned synthetic pressure field."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    base = torch.sin(2.5 * x) + 0.7 * torch.cos(3.0 * y) + 0.35 * z
    interaction = 0.25 * x * y - 0.12 * y * z
    radial = torch.sqrt(x * x + y * y + z * z + 1e-6)

    if topology == "toroidal":
        return base + interaction + 0.15 * torch.sin(4.0 * radial)
    if topology == "spherical":
        return base + 0.20 * torch.cos(5.0 * radial) - 0.08 * x * z
    # volumetric and high-genus/open cases
    return base + 0.18 * torch.sin(2.0 * x * z) + 0.1 * radial


def apply_complex_deformation(points, generator, case_name):
    """Add nonlinear deformations to make mixed-geometry task harder."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    noise = torch.randn(points.shape, generator=generator, device=points.device) * 0.03
    twist = torch.rand(1, generator=generator, device=points.device).item() * 0.6 + 0.2
    bend = torch.rand(1, generator=generator, device=points.device).item() * 0.5 + 0.1

    if case_name == "spherical":
        r = torch.sqrt(x * x + y * y + z * z + 1e-6)
        points[:, 0] = x + 0.12 * torch.sin(3.0 * y) + 0.08 * r * torch.cos(2.0 * z)
        points[:, 1] = y + 0.10 * torch.sin(2.5 * z) - 0.06 * r * torch.sin(2.0 * x)
        points[:, 2] = z + 0.07 * torch.cos(3.5 * x)
    elif case_name == "toroidal":
        theta = torch.atan2(y, x + 1e-6)
        points[:, 0] = x + twist * 0.08 * torch.sin(4.0 * theta) + 0.06 * torch.cos(2.0 * z)
        points[:, 1] = y + twist * 0.08 * torch.cos(3.0 * theta) - 0.04 * torch.sin(2.0 * z)
        points[:, 2] = z + 0.09 * torch.sin(3.0 * theta) + 0.05 * torch.cos(2.0 * x)
    elif case_name == "open_surface":
        points[:, 0] = x + 0.18 * torch.sin(2.2 * y) + 0.1 * x * z
        points[:, 1] = y + bend * 0.14 * torch.sin(2.0 * x) - 0.08 * z
        points[:, 2] = z + 0.15 * torch.cos(2.8 * y) + 0.04 * x * y
    else:  # high_genus
        points[:, 0] = x + 0.2 * torch.sin(3.3 * y) + 0.12 * torch.cos(2.3 * z)
        points[:, 1] = y + 0.18 * torch.sin(2.7 * x) + 0.10 * torch.sin(2.1 * z)
        points[:, 2] = z + 0.16 * torch.cos(3.1 * x) - 0.08 * x * y

    points += noise
    return points


def save_loss_plots(train_losses, test_losses, train_topo_hist, test_topo_hist, out_path_prefix):
    """Save overall and per-topology loss curves."""
    os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train L2")
    plt.plot(epochs, test_losses, label="Test L2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TOPOS Online Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_overall.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for topo in sorted(test_topo_hist.keys()):
        if len(test_topo_hist[topo]) == len(epochs):
            plt.plot(epochs, test_topo_hist[topo], label=f"Test-{topo}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TOPOS Test Loss by Source Topology")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_per_topology.png", dpi=160)
    plt.close()


class SyntheticGeometryDatasetTOPOS(Dataset):
    """
    Dynamically generates the training dataset entirely from scratch by:
    1. Using Latent Geometry Generators to create synthetic 3D "physical" geometries.
    2. Deforming them sequentially to simulate real variations.
    3. Mapping them back to the pure latent topological grid via OT3Dto2DMapper.
    """
    def __init__(
        self,
        cache_dir=None,
        n_train=500,
        n_test=111,
        split='train',
        expand_factor=2.0,
        num_points=3586,
        base_seed=42,
    ):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.expand_factor = expand_factor
        self.num_points = num_points
        self.base_seed = base_seed
        self.router = TopologicalRouter()
        
        self.num_samples = n_train if split == 'train' else n_test
        self.split = split
        print(f"[*] SyntheticGeometryDataset (Online TOPOS) [{split}] initialized with {self.num_samples} generated samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == 'train' else idx + 1000
        sample_seed = self.base_seed + file_idx
        
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_topos_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"points", "normals", "pressure", "topology", "chi", "idx_encoder", "idx_decoder", "latent_coords", "latent_normals", "grid_width", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached
                
        # --- 1. GENERATE PHYSICAL DATA FROM MIXED-GENUS CASE LIBRARY ---
        case = CASE_LIBRARY[idx % len(CASE_LIBRARY)]
        topology = case["name"]
        chi = case["chi"]
        latent_topology = case["latent_topology"]
        
        dummy_mapper = OT3Dto2DMapper(
            latent_topology=latent_topology,
            expand_factor=self.expand_factor,
            width=84 if latent_topology != "volumetric" else 16,
        )
        
        # Pull structural geometry directly from the proper geometry generator
        if latent_topology == "toroidal":
            clean_points, _ = dummy_mapper._generate_latent_torus(self.num_points)
        elif latent_topology == "spherical":
            clean_points, _ = dummy_mapper._generate_latent_sphere(self.num_points)
        else: # volumetric
            clean_points, _ = dummy_mapper._generate_latent_volume(self.num_points)

        # Generator must match tensor device type for torch.rand* calls.
        g = torch.Generator(device=clean_points.device.type)
        g.manual_seed(sample_seed)
            
        # Deform the generation
        clean_points = sample_or_repeat_points(clean_points, self.num_points, g)

        scale_x = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
        scale_y = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
        scale_z = torch.rand(1, generator=g, device=clean_points.device) * 0.3 + 0.85
        
        points = clean_points.clone()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        points[:, 2] *= scale_z
        points = apply_complex_deformation(points, g, topology)
        points = points.float()
        
        normals = points / (torch.linalg.norm(points, dim=-1, keepdim=True) + 1e-6)
        pressure = synthetic_pressure(points, topology)

        # --- 2. MAP TO LATENT 2D GEOMETRY ---
        scaled_width = 84
        if latent_topology == "volumetric":
            scaled_width = 16 
        
        mapper = OT3Dto2DMapper(latent_topology=latent_topology, expand_factor=self.expand_factor, width=scaled_width)
        idx_encoder, idx_decoder, grid_width = mapper.get_otno_indices(points, blur=0.01)
        
        # --- 3. EXPLICIT LATENT STRUCTURAL DOMAIN ---
        if latent_topology == "toroidal":
            latent_coords, _ = mapper._generate_latent_torus(self.num_points)
            latent_coords = latent_coords.view(grid_width, grid_width, 3)
            latent_normals = compute_torus_normals(grid_width)
        elif latent_topology == "spherical":
            latent_coords, _ = mapper._generate_latent_sphere(self.num_points)
            latent_coords = latent_coords.view(grid_width, grid_width, 3)
            latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)
        else:
            latent_coords, _ = mapper._generate_latent_volume(self.num_points)
            latent_coords = latent_coords.view(grid_width, grid_width, grid_width, 3)
            latent_normals = latent_coords / (torch.linalg.norm(latent_coords, dim=-1, keepdim=True) + 1e-6)

        data_dict = {
            'points': points.cpu(),
            'normals': normals.cpu(),
            'pressure': pressure.cpu(),
            'topology': topology,
            'source_latent_topology': latent_topology,
            'chi': chi,
            'idx_encoder': idx_encoder.long().cpu(),
            'idx_decoder': idx_decoder.long().cpu(),
            'latent_coords': latent_coords.cpu(),
            'latent_normals': latent_normals.cpu(),
            'grid_width': grid_width,
            'schema_version': SYNTH_SCHEMA_VERSION,
        }
        
        if self.cache_dir:
            torch.save(data_dict, cache_path)
            
        return data_dict


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TOPOS Unified Trainer for Synthetic Generation")
    parser.add_argument('--config', type=str, default='configs/mixed_genus_fair_comparison.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='otno-topos-mixed-geometry')
    parser.add_argument('--wandb_entity', type=str, default=os.environ.get('WANDB_ENTITY', 'mamtapc003-zenteiq-ai'))
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--gpus', type=str, default='auto', help='GPU selection: "auto", "cpu", "0", "1", "0,1", or "all"')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    set_seed(config['training'].get('seed', 42))
    device, gpu_ids = resolve_device(args.gpus)
    print(f"[*] Using device {device}" + (f" with GPUs {gpu_ids}" if gpu_ids else ""))
    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("[!] wandb requested but not installed. Continuing without wandb logging.")
        else:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config=config,
                    tags=["topos", "online", "mixed-genus", "router"],
                )
            except Exception as e:
                print(f"[!] wandb init failed: {e}. Continuing without wandb logging.")
                wandb_run = None
    
    cache_dir = config['dataset'].get('cache_dir', None)
    if cache_dir:
        cache_dir = os.path.join(cache_dir, 'synthetic')

    train_ds = SyntheticGeometryDatasetTOPOS(cache_dir=cache_dir,
                                     n_train=config['dataset']['train_samples'], 
                                     n_test=config['dataset']['test_samples'],
                                     split='train', 
                                     expand_factor=config['dataset']['expand_factor'],
                                     num_points=config['dataset'].get('num_points', 3586),
                                     base_seed=config['training'].get('seed', 42))
                                     
    test_ds = SyntheticGeometryDatasetTOPOS(cache_dir=cache_dir,
                                    n_train=config['dataset']['train_samples'], 
                                    n_test=config['dataset']['test_samples'],
                                    split='test', 
                                    expand_factor=config['dataset']['expand_factor'],
                                    num_points=config['dataset'].get('num_points', 3586),
                                    base_seed=config['training'].get('seed', 42))

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    # ---------------------------------------------------------
    # TOPOS Model Initialization
    # ---------------------------------------------------------
    if 'spherical_config' in config['model']:
        spherical_config = config['model']['spherical_config']
        volumetric_config = config['model']['volumetric_config']
        toroidal_config = config['model'].get('toroidal_config', None)
        spherical_config['n_modes'] = tuple(spherical_config['n_modes'])
        if volumetric_config:
            volumetric_config['n_modes'] = tuple(volumetric_config['n_modes'])
        if toroidal_config:
            toroidal_config['n_modes'] = tuple(toroidal_config['n_modes'])
    else:
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
        volumetric_config = model_config.copy()
        volumetric_config['n_modes'] = (model_config['n_modes'][0]//2, model_config['n_modes'][1]//2, 1)

    graph_config = {
        'in_channels': config['model']['in_channels'],
        'out_channels': config['model']['out_channels'],
        'hidden_channels': config['model']['hidden_channels'] // 3
    }

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=toroidal_config,
        volumetric_config=volumetric_config,
        graph_config=graph_config
    )
    model, device, gpu_ids = prepare_model_for_devices(model, args.gpus)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    myloss = LpLoss(size_average=False)

    # ---------------------------------------------------------
    # Normalization (shared protocol with OTNO)
    # ---------------------------------------------------------
    print("[*] Estimating normalization statistics over dynamic dataset (up to 50 samples)...")
    pressures_list = []
    topo_transports = defaultdict(list)
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        points = batch['points'][0]
        normals = batch['normals'][0]
        topology = batch['topology'][0]
        idx_encoder = batch['idx_encoder'][0]
        grid_width = batch['grid_width'][0].item()
        latent_coords = batch['latent_coords'][0]
        latent_normals = batch['latent_normals'][0]

        mapped_phys_points = points[idx_encoder]
        mapped_phys_normals = normals[idx_encoder]
        normal_cross = torch.cross(mapped_phys_normals, latent_normals.view(-1, 3), dim=-1)

        is_volumetric = latent_coords.dim() == 4
        if is_volumetric:
            view_shape = (grid_width, grid_width, grid_width, 3)
            permute_shape = (3, 0, 1, 2)
        else:
            view_shape = (grid_width, grid_width, 3)
            permute_shape = (2, 0, 1)

        mapped_phys_points = mapped_phys_points.view(view_shape)
        normal_cross = normal_cross.view(view_shape)
        transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(permute_shape).unsqueeze(0)

        topo_transports[topology].append(transports.cpu())
        pressures_list.append(batch['pressure'][0].cpu())

    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures_list, dim=0), reduce_dim=[0])
    pressure_norm.to(device)

    transport_norm = {}
    for topology, samples in topo_transports.items():
        if len(samples) == 0:
            continue
        # samples for each topology share shape (2D or 3D), so they can be concatenated safely.
        reduce_dim = [0, 2, 3, 4] if samples[0].dim() == 5 else [0, 2, 3]
        norm = UnitGaussianNormalizer(torch.cat(samples, dim=0), reduce_dim=reduce_dim)
        norm.to(device)
        transport_norm[topology] = norm

    print(f"[*] Normalizers initialized. Pressure mean={pressure_norm.mean.item():.4f}, std={pressure_norm.std.item():.4f}")

    # ---------------------------------------------------------
    # TOPOS Training Loop
    # ---------------------------------------------------------
    epochs = config['training']['n_epochs']
    print(f"[*] Training TOPOS Online for {epochs} epochs...")
    train_losses = []
    test_losses = []
    train_topo_history = defaultdict(list)
    test_topo_history = defaultdict(list)
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_l2 = 0.0
        train_topo_loss = defaultdict(float)
        train_topo_count = defaultdict(int)
        routed_count = defaultdict(int)
        
        for batch in train_loader:
            optimizer.zero_grad()
            points = batch['points'][0].to(device)
            normals = batch['normals'][0].to(device)
            pressure = batch['pressure'][0].to(device)
            topology = batch['topology'][0]
            chi = batch['chi'][0].item()
            routed_topology = model.route(chi=chi)
            routed_count[routed_topology] += 1
            
            # Construct Dynamic Target Field
            if topology == "graph":
                out = model(points=points.unsqueeze(0),
                            features=torch.cat([points, normals, torch.cross(points,normals,dim=1)],dim=1).unsqueeze(0),
                            topology="auto", chi=chi)
                pressure_normed = pressure_norm.encode(pressure.clone().unsqueeze(0)).squeeze(0)
                loss = myloss(out.view(1, -1), pressure_normed.view(1, -1))
            else:
                idx_encoder = batch['idx_encoder'][0].to(device)
                idx_decoder = batch['idx_decoder'][0].to(device)
                grid_width = batch['grid_width'][0].item()
                latent_coords = batch['latent_coords'][0].to(device)
                latent_normals = batch['latent_normals'][0].to(device)

                # Construct 9-channel dynamically mapping back points
                mapped_phys_points = points[idx_encoder]
                mapped_phys_normals = normals[idx_encoder]
                normal_cross = torch.cross(mapped_phys_normals, latent_normals.view(-1, 3), dim=-1)

                is_volumetric = latent_coords.dim() == 4
                if is_volumetric:
                    view_shape = (grid_width, grid_width, grid_width, 3)
                    permute_shape = (3, 0, 1, 2)
                else: # Spherical or Toroidal
                    view_shape = (grid_width, grid_width, 3)
                    permute_shape = (2, 0, 1)

                mapped_phys_points = mapped_phys_points.view(view_shape)
                normal_cross = normal_cross.view(view_shape)
                
                transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(permute_shape).unsqueeze(0)
                if topology in transport_norm:
                    transports = transport_norm[topology].encode(transports.clone())
                
                pressure_normed = pressure_norm.encode(pressure.clone().unsqueeze(0)).squeeze(0)
                out = model(transports=transports, idx_decoder=idx_decoder, topology="auto", chi=chi)
                loss = myloss(out.view(1, -1), pressure_normed.view(1, -1))
            
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
            train_topo_loss[topology] += loss.item()
            train_topo_count[topology] += 1

        scheduler.step()

        # Evaluate
        model.eval()
        test_l2 = 0.0
        test_topo_loss = defaultdict(float)
        test_topo_count = defaultdict(int)
        test_routed_count = defaultdict(int)
        with torch.no_grad():
            for batch in test_loader:
                points = batch['points'][0].to(device)
                normals = batch['normals'][0].to(device)
                pressure = batch['pressure'][0].to(device)
                topology = batch['topology'][0]
                chi = batch['chi'][0].item()
                routed_topology = model.route(chi=chi)
                test_routed_count[routed_topology] += 1

                if topology == "graph":
                    out = model(points=points.unsqueeze(0),
                                features=torch.cat([points, normals, torch.cross(points,normals,dim=1)],dim=1).unsqueeze(0),
                                topology="auto", chi=chi)
                    out = pressure_norm.decode(out.clone())
                else:
                    idx_encoder = batch['idx_encoder'][0].to(device)
                    idx_decoder = batch['idx_decoder'][0].to(device)
                    grid_width = batch['grid_width'][0].item()
                    latent_coords = batch['latent_coords'][0].to(device)
                    latent_normals = batch['latent_normals'][0].to(device)

                    mapped_phys_points = points[idx_encoder]
                    mapped_phys_normals = normals[idx_encoder]
                    normal_cross = torch.cross(mapped_phys_normals, latent_normals.view(-1, 3), dim=-1)

                    is_volumetric = latent_coords.dim() == 4
                    if is_volumetric:
                        view_shape = (grid_width, grid_width, grid_width, 3)
                        permute_shape = (3, 0, 1, 2)
                    else:
                        view_shape = (grid_width, grid_width, 3)
                        permute_shape = (2, 0, 1)

                    mapped_phys_points = mapped_phys_points.view(view_shape)
                    normal_cross = normal_cross.view(view_shape)
                    transports = torch.cat([mapped_phys_points, latent_coords, normal_cross], dim=-1).permute(permute_shape).unsqueeze(0)
                    if topology in transport_norm:
                        transports = transport_norm[topology].encode(transports.clone())
                    
                    out = model(transports=transports, idx_decoder=idx_decoder, topology="auto", chi=chi)
                    out = pressure_norm.decode(out.clone())

                loss_val = myloss(out.view(1, -1), pressure.view(1, -1)).item()
                test_l2 += loss_val
                test_topo_loss[topology] += loss_val
                test_topo_count[topology] += 1

        train_l2 /= len(train_ds)
        test_l2 /= len(test_ds)
        train_losses.append(train_l2)
        test_losses.append(test_l2)
        for topo in sorted(set(list(train_topo_count.keys()) + list(test_topo_count.keys()))):
            train_topo_history[topo].append(train_topo_loss[topo] / max(train_topo_count[topo], 1))
            test_topo_history[topo].append(test_topo_loss[topo] / max(test_topo_count[topo], 1))
        train_parts = [f"{k}:{(train_topo_loss[k]/max(train_topo_count[k],1)):.4f}" for k in sorted(train_topo_count.keys())]
        test_parts = [f"{k}:{(test_topo_loss[k]/max(test_topo_count[k],1)):.4f}" for k in sorted(test_topo_count.keys())]
        routed_train_parts = [f"{k}:{v}" for k, v in sorted(routed_count.items())]
        routed_test_parts = [f"{k}:{v}" for k, v in sorted(test_routed_count.items())]
        print(
            f"Epoch {ep+1}/{epochs}, Train L2: {train_l2:.6f}, Test L2: {test_l2:.6f}, "
            f"TrainTopo[{', '.join(train_parts)}], TestTopo[{', '.join(test_parts)}], "
            f"RouterTrain[{', '.join(routed_train_parts)}], RouterTest[{', '.join(routed_test_parts)}], "
            f"Time: {default_timer()-t1:.2f}s"
        )
        if wandb_run is not None:
            metrics = {
                "epoch": ep + 1,
                "train/l2": train_l2,
                "test/l2": test_l2,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            for topo in sorted(train_topo_count.keys()):
                metrics[f"train/topology_{topo}"] = train_topo_history[topo][-1]
            for topo in sorted(test_topo_count.keys()):
                metrics[f"test/topology_{topo}"] = test_topo_history[topo][-1]
            for route, count in sorted(routed_count.items()):
                metrics[f"router/train_{route}"] = count
            for route, count in sorted(test_routed_count.items()):
                metrics[f"router/test_{route}"] = count
            wandb_run.log(metrics, step=ep + 1)

    out_dir = config.get("output", {}).get("dir", "results")
    
    chk_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, "topos_mixed_genus.pt")
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(model_state, chk_path)
    print(f"[*] Saved TOPOS weights to: {chk_path}")

    plot_prefix = os.path.join(out_dir, "topos_online_mixed")
    save_loss_plots(train_losses, test_losses, train_topo_history, test_topo_history, plot_prefix)
    print(f"[*] Saved TOPOS loss plots: {plot_prefix}_overall.png and {plot_prefix}_per_topology.png")
    if wandb_run is not None:
        overall_plot = f"{plot_prefix}_overall.png"
        topo_plot = f"{plot_prefix}_per_topology.png"
        if os.path.exists(overall_plot):
            wandb_run.log({"plots/overall": wandb.Image(overall_plot)})
        if os.path.exists(topo_plot):
            wandb_run.log({"plots/per_topology": wandb.Image(topo_plot)})
        wandb_run.finish()

if __name__ == "__main__":
    main()
