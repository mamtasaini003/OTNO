import os
import sys
import yaml
import argparse
import random
from collections import defaultdict
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topos.models.baselines import UFNO
from topos.data.ot_mapper_3d import OT3Dto2DMapper
from topos.utils import LpLoss, UnitGaussianNormalizer

SYNTH_SCHEMA_VERSION = 1
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
    phi = torch.linspace(0, 2 * np.pi, width + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta = r * torch.cos(theta)
    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi = (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi = torch.zeros_like(dx_dphi)
    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi
    normals = torch.stack((nx, ny, nz), dim=-1)
    normals = normals / torch.linalg.norm(normals, dim=2, keepdim=True)
    return normals


def sample_or_repeat_points(points, target_n, generator):
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
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    base = torch.sin(2.5 * x) + 0.7 * torch.cos(3.0 * y) + 0.35 * z
    interaction = 0.25 * x * y - 0.12 * y * z
    radial = torch.sqrt(x * x + y * y + z * z + 1e-6)
    if topology == "toroidal":
        return base + interaction + 0.15 * torch.sin(4.0 * radial)
    if topology == "spherical":
        return base + 0.20 * torch.cos(5.0 * radial) - 0.08 * x * z
    return base + 0.18 * torch.sin(2.0 * x * z) + 0.1 * radial


def apply_complex_deformation(points, generator, case_name):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    noise = torch.randn(points.shape, generator=generator, device=points.device) * 0.03
    if case_name == "spherical":
        r = torch.sqrt(x * x + y * y + z * z + 1e-6)
        points[:, 0] = x + 0.12 * torch.sin(3.0 * y) + 0.08 * r * torch.cos(2.0 * z)
        points[:, 1] = y + 0.10 * torch.sin(2.5 * z) - 0.06 * r * torch.sin(2.0 * x)
        points[:, 2] = z + 0.07 * torch.cos(3.5 * x)
    elif case_name == "toroidal":
        theta = torch.atan2(y, x + 1e-6)
        points[:, 0] = x + 0.08 * torch.sin(4.0 * theta) + 0.06 * torch.cos(2.0 * z)
        points[:, 1] = y + 0.08 * torch.cos(3.0 * theta) - 0.04 * torch.sin(2.0 * z)
        points[:, 2] = z + 0.09 * torch.sin(3.0 * theta) + 0.05 * torch.cos(2.0 * x)
    elif case_name == "open_surface":
        points[:, 0] = x + 0.18 * torch.sin(2.2 * y) + 0.1 * x * z
        points[:, 1] = y + 0.14 * torch.sin(2.0 * x) - 0.08 * z
        points[:, 2] = z + 0.15 * torch.cos(2.8 * y) + 0.04 * x * y
    else:
        points[:, 0] = x + 0.2 * torch.sin(3.3 * y) + 0.12 * torch.cos(2.3 * z)
        points[:, 1] = y + 0.18 * torch.sin(2.7 * x) + 0.10 * torch.sin(2.1 * z)
        points[:, 2] = z + 0.16 * torch.cos(3.1 * x) - 0.08 * x * y
    points += noise
    return points


def save_loss_plots(train_losses, test_losses, test_topology_history, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train L2")
    plt.plot(epochs, test_losses, label="Test L2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UFNO Online Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_overall.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for topo in sorted(test_topology_history.keys()):
        vals = test_topology_history[topo]
        if len(vals) == len(epochs):
            plt.plot(epochs, vals, label=f"Test-{topo}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UFNO Test Loss by Source Topology")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_per_topology.png", dpi=160)
    plt.close()


class MixedGenus2DLatentDataset(Dataset):
    """Shared mixed-genus source generation with OT mapping to a 2D torus latent grid."""

    def __init__(self, cache_dir=None, n_train=500, n_test=111, split="train", expand_factor=2.0, num_points=3586, base_seed=42):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.num_samples = n_train if split == "train" else n_test
        self.split = split
        self.num_points = num_points
        self.base_seed = base_seed
        self.expand_factor = expand_factor
        self.mapper = OT3Dto2DMapper(latent_topology="toroidal", expand_factor=expand_factor, width=84)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = idx if self.split == "train" else idx + 1000
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"synthetic_ufno_{self.split}_{file_idx:03d}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=False)
                required = {"transports", "pressure", "idx_decoder", "source_topology", "schema_version"}
                if required.issubset(cached.keys()) and cached.get("schema_version") == SYNTH_SCHEMA_VERSION:
                    return cached

        sample_seed = self.base_seed + file_idx
        case = CASE_LIBRARY[idx % len(CASE_LIBRARY)]
        topology = case["name"]
        latent_topology = case["latent_topology"]

        dummy_mapper = OT3Dto2DMapper(
            latent_topology=latent_topology,
            expand_factor=self.expand_factor,
            width=84 if latent_topology != "volumetric" else 16,
        )
        if latent_topology == "toroidal":
            clean_points, _ = dummy_mapper._generate_latent_torus(self.num_points)
        elif latent_topology == "spherical":
            clean_points, _ = dummy_mapper._generate_latent_sphere(self.num_points)
        else:
            clean_points, _ = dummy_mapper._generate_latent_volume(self.num_points)

        g = torch.Generator(device=clean_points.device.type)
        g.manual_seed(sample_seed)
        clean_points = sample_or_repeat_points(clean_points, self.num_points, g)

        sx = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
        sy = torch.rand(1, generator=g, device=clean_points.device) * 0.4 + 0.8
        sz = torch.rand(1, generator=g, device=clean_points.device) * 0.3 + 0.85
        points = clean_points.clone()
        points[:, 0] *= sx
        points[:, 1] *= sy
        points[:, 2] *= sz
        points = apply_complex_deformation(points, g, topology).float()
        normals = points / (torch.linalg.norm(points, dim=-1, keepdim=True) + 1e-6)
        pressure = synthetic_pressure(points, topology)

        idx_encoder, idx_decoder, width = self.mapper.get_otno_indices(points, blur=0.01)
        latent_coords, _ = self.mapper._generate_latent_torus(self.num_points)
        latent_coords = latent_coords.view(width, width, 3).float()
        latent_normals = compute_torus_normals(width).float()
        mapped_points = points[idx_encoder].view(width, width, 3)
        mapped_normals = normals[idx_encoder].view(width, width, 3)
        normal_cross = torch.cross(mapped_normals, latent_normals, dim=-1)
        transports = torch.cat([mapped_points, latent_coords, normal_cross], dim=-1).permute(2, 0, 1)

        out = {
            "transports": transports.cpu(),
            "pressure": pressure.cpu(),
            "idx_decoder": idx_decoder.long().cpu(),
            "source_topology": topology,
            "schema_version": SYNTH_SCHEMA_VERSION,
        }
        if self.cache_dir:
            torch.save(out, cache_path)
        return out


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="UFNO Online Trainer (Fair Mixed-Genus Comparison)")
    parser.add_argument("--config", type=str, default="configs/abc_comparison.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="otno-topos-mixed-geometry")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", "mamtapc003-zenteiq-ai"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    set_seed(config["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                    tags=["ufno", "online", "mixed-genus"],
                )
            except Exception as e:
                print(f"[!] wandb init failed: {e}. Continuing without wandb logging.")
                wandb_run = None
    cache_dir = config["dataset"].get("cache_dir", None)
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    train_ds = MixedGenus2DLatentDataset(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="train",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
    )
    test_ds = MixedGenus2DLatentDataset(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    pressures = []
    transports = []
    for i, b in enumerate(train_loader):
        if i >= 50:
            break
        pressures.append(b["pressure"][0])
        transports.append(b["transports"])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, dim=0), reduce_dim=[0]).to(device)
    transport_norm = UnitGaussianNormalizer(torch.cat(transports, dim=0), reduce_dim=[0, 2, 3]).to(device)

    model = UFNO(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        n_modes=tuple(config["model"]["n_modes"]),
        hidden_channels=config["model"]["hidden_channels"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["step_size"], gamma=config["training"]["gamma"])
    loss_fn = LpLoss(size_average=False)

    train_losses, test_losses = [], []
    test_topo_history = defaultdict(list)
    epochs = config["training"]["n_epochs"]
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        tr = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            x = transport_norm.encode(b["transports"].to(device).clone())
            y = pressure_norm.encode(b["pressure"].to(device).clone())
            idx = b["idx_decoder"][0].to(device)
            out = model(x)
            out = out.reshape(out.shape[0], out.shape[1], -1)[:, :, idx]
            loss = loss_fn(out, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            tr += loss.item()
        scheduler.step()

        model.eval()
        te = 0.0
        topo_loss = defaultdict(float)
        topo_count = defaultdict(int)
        with torch.no_grad():
            for b in test_loader:
                x = transport_norm.encode(b["transports"].to(device).clone())
                y = b["pressure"].to(device)
                idx = b["idx_decoder"][0].to(device)
                topo = b["source_topology"][0]
                out = model(x).reshape(x.shape[0], 1, -1)[:, :, idx]
                out = pressure_norm.decode(out.clone())
                lv = loss_fn(out, y.unsqueeze(1)).item()
                te += lv
                topo_loss[topo] += lv
                topo_count[topo] += 1
        tr /= len(train_ds)
        te /= len(test_ds)
        train_losses.append(tr)
        test_losses.append(te)
        for topo in sorted(topo_count.keys()):
            test_topo_history[topo].append(topo_loss[topo] / max(topo_count[topo], 1))
        topo_parts = [f"{k}:{test_topo_history[k][-1]:.4f}" for k in sorted(topo_count.keys())]
        print(f"Epoch {ep+1}/{epochs}, Train L2: {tr:.6f}, Test L2: {te:.6f}, TestSrc[{', '.join(topo_parts)}], Time: {default_timer()-t1:.2f}s")
        if wandb_run is not None:
            metrics = {
                "epoch": ep + 1,
                "train/l2": tr,
                "test/l2": te,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            for topo in sorted(topo_count.keys()):
                metrics[f"test/topology_{topo}"] = test_topo_history[topo][-1]
            wandb_run.log(metrics, step=ep + 1)

    out_dir = config.get("output", {}).get("dir", "results")
    prefix = os.path.join(out_dir, "ufno_online_mixed")
    save_loss_plots(train_losses, test_losses, test_topo_history, prefix)
    print(f"[*] Saved UFNO loss plots: {prefix}_overall.png and {prefix}_per_topology.png")
    if wandb_run is not None:
        overall_plot = f"{prefix}_overall.png"
        topo_plot = f"{prefix}_per_topology.png"
        if os.path.exists(overall_plot):
            wandb_run.log({"plots/overall": wandb.Image(overall_plot)})
        if os.path.exists(topo_plot):
            wandb_run.log({"plots/per_topology": wandb.Image(topo_plot)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
