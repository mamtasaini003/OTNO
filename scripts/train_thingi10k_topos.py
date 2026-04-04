import argparse
import os
import random
import sys
from collections import defaultdict
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topos.data.thingi10k_geometry import Thingi10KToposDataset, parse_supported_topologies
from topos.models import TOPOS
from topos.utils import LpLoss, UnitGaussianNormalizer, prepare_model_for_devices, resolve_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_loss_plots(train_losses, test_losses, test_topology_history, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train L2")
    plt.plot(epochs, test_losses, label="Test L2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Thingi10K TOPOS Loss Curves")
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
    plt.title("Thingi10K TOPOS Test Loss by Source Topology")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_per_topology.png", dpi=160)
    plt.close()


def build_datasets(config):
    dataset_cfg = config["dataset"]
    cache_dir = dataset_cfg.get("cache_dir")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "thingi10k")

    # --- Topology filtering ---
    topo_file = dataset_cfg.get("supported_topologies_file")
    supported_topologies = None
    if topo_file:
        supported_topologies = parse_supported_topologies(topo_file)
        print(f"[*] Filtering to supported topologies from {topo_file}: {sorted(supported_topologies)}")
    else:
        print("[*] Using all geometries (no topology filter).")

    common = dict(
        cache_dir=cache_dir,
        train_samples=dataset_cfg["train_samples"],
        test_samples=dataset_cfg["test_samples"],
        split_offset=dataset_cfg.get("split_offset", 0),
        variant=dataset_cfg.get("variant", "npz"),
        source_cache_dir=dataset_cfg.get("source_cache_dir"),
        num_points=dataset_cfg.get("num_points", 3586),
        seed=config["training"].get("seed", 42),
        thingi_filters=dataset_cfg.get("filters", {}),
        expand_factor=dataset_cfg.get("expand_factor", 2.0),
        width_2d=dataset_cfg.get("resolution_2d", 84),
        width_3d=dataset_cfg.get("resolution_3d", 16),
        supported_topologies=supported_topologies,
    )
    train_ds = Thingi10KToposDataset(split="train", **common)
    test_ds = Thingi10KToposDataset(split="test", **common)
    return train_ds, test_ds


def build_transports(batch, device):
    topology = batch["topology"][0]
    if topology == "graph":
        return None

    points = batch["points"][0].to(device)
    normals = batch["normals"][0].to(device)
    idx_encoder = batch["idx_encoder"][0].to(device)
    grid_width = batch["grid_width"][0].item()
    latent_coords = batch["latent_coords"][0].to(device)
    latent_normals = batch["latent_normals"][0].to(device)

    mapped_points = points[idx_encoder]
    mapped_normals = normals[idx_encoder]
    normal_cross = torch.cross(mapped_normals, latent_normals.view(-1, 3), dim=-1)

    is_volumetric = latent_coords.dim() == 4
    if is_volumetric:
        view_shape = (grid_width, grid_width, grid_width, 3)
        permute_shape = (3, 0, 1, 2)
    else:
        view_shape = (grid_width, grid_width, 3)
        permute_shape = (2, 0, 1)

    mapped_points = mapped_points.view(view_shape)
    normal_cross = normal_cross.view(view_shape)
    transports = torch.cat([mapped_points, latent_coords, normal_cross], dim=-1).permute(permute_shape).unsqueeze(0)
    return transports


def main():
    parser = argparse.ArgumentParser(description="TOPOS Trainer for Thingi10K")
    parser.add_argument("--config", type=str, default="configs/thingi10k_topos.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="otno-topos-thingi10k")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    set_seed(config["training"].get("seed", 42))
    device, gpu_ids = resolve_device(args.gpus)
    print(f"[*] Using device {device}" + (f" with GPUs {gpu_ids}" if gpu_ids else ""))

    wandb_run = None
    if args.wandb and wandb is not None:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=config,
                tags=["thingi10k", "topos", "router"],
            )
        except Exception as exc:
            print(f"[!] wandb init failed: {exc}")

    train_ds, test_ds = build_datasets(config)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    model_cfg = config["model"]
    if "spherical_config" in model_cfg:
        spherical_config = model_cfg["spherical_config"]
        toroidal_config = model_cfg.get("toroidal_config")
        volumetric_config = model_cfg.get("volumetric_config")
        spherical_config["n_modes"] = tuple(spherical_config["n_modes"])
        if toroidal_config is not None:
            toroidal_config["n_modes"] = tuple(toroidal_config["n_modes"])
        if volumetric_config is not None:
            volumetric_config["n_modes"] = tuple(volumetric_config["n_modes"])
    else:
        shared = {
            "n_modes": tuple(model_cfg["n_modes"]),
            "hidden_channels": model_cfg["hidden_channels"],
            "in_channels": model_cfg["in_channels"],
            "out_channels": model_cfg["out_channels"],
            "n_layers": model_cfg["n_layers"],
            "use_mlp": model_cfg["use_mlp"],
            "mlp": model_cfg["mlp"],
            "norm": model_cfg.get("norm"),
            "domain_padding": model_cfg.get("domain_padding"),
            "factorization": model_cfg.get("factorization"),
            "rank": model_cfg.get("rank", 1.0),
        }
        spherical_config = shared.copy()
        toroidal_config = shared.copy()
        volumetric_config = shared.copy()
        volumetric_config["n_modes"] = tuple(model_cfg.get("volumetric_n_modes", [8, 8, 1]))
    graph_config = model_cfg.get(
        "graph_config",
        {"in_channels": model_cfg["in_channels"], "out_channels": model_cfg["out_channels"], "hidden_channels": model_cfg["hidden_channels"] // 3},
    )

    model = TOPOS(
        spherical_config=spherical_config,
        toroidal_config=toroidal_config,
        volumetric_config=volumetric_config,
        graph_config=graph_config,
    )
    model, device, gpu_ids = prepare_model_for_devices(model, args.gpus)

    topo_transports = defaultdict(list)
    pressure_batches = []
    for i, batch in enumerate(train_loader):
        if i >= min(50, len(train_loader)):
            break
        transports = build_transports(batch, torch.device("cpu"))
        if transports is not None:
            topo_transports[batch["topology"][0]].append(transports)
        pressure_batches.append(batch["pressure"][0].unsqueeze(0))
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressure_batches, dim=0), reduce_dim=[0, 1]).to(device)
    transport_norm = {}
    for topo, samples in topo_transports.items():
        dims = [0, 2, 3, 4] if samples[0].dim() == 5 else [0, 2, 3]
        transport_norm[topo] = UnitGaussianNormalizer(torch.cat(samples, dim=0), reduce_dim=dims).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["step_size"], gamma=config["training"]["gamma"])
    loss_fn = LpLoss(size_average=False)

    train_losses, test_losses = [], []
    test_topo_history = defaultdict(list)
    for ep in range(config["training"]["n_epochs"]):
        t1 = default_timer()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            topology = batch["topology"][0]
            chi = float(batch["chi"][0].item())
            target = pressure_norm.encode(batch["pressure"].to(device).clone())

            if topology == "graph":
                pred = model(points=batch["points"].to(device), features=torch.cat([batch["points"], batch["normals"], torch.cross(batch["points"], batch["normals"], dim=-1)], dim=-1).to(device), topology=topology, chi=chi)
            else:
                transports = build_transports(batch, device)
                if topology in transport_norm:
                    transports = transport_norm[topology].encode(transports.clone())
                pred = model(transports=transports, idx_decoder=batch["idx_decoder"][0].to(device), topology=topology, chi=chi)

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        test_loss = 0.0
        topo_loss = defaultdict(float)
        topo_count = defaultdict(int)
        with torch.no_grad():
            for batch in test_loader:
                topology = batch["topology"][0]
                chi = float(batch["chi"][0].item())
                y = batch["pressure"].to(device)
                if topology == "graph":
                    pred = model(points=batch["points"].to(device), features=torch.cat([batch["points"], batch["normals"], torch.cross(batch["points"], batch["normals"], dim=-1)], dim=-1).to(device), topology=topology, chi=chi)
                else:
                    transports = build_transports(batch, device)
                    if topology in transport_norm:
                        transports = transport_norm[topology].encode(transports.clone())
                    pred = model(transports=transports, idx_decoder=batch["idx_decoder"][0].to(device), topology=topology, chi=chi)
                pred = pressure_norm.decode(pred.clone())
                loss_value = loss_fn(pred, y).item()
                test_loss += loss_value
                topo_loss[topology] += loss_value
                topo_count[topology] += 1

        train_loss /= len(train_ds)
        test_loss /= len(test_ds)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        for topo in sorted(topo_count):
            test_topo_history[topo].append(topo_loss[topo] / max(topo_count[topo], 1))
        topo_parts = [f"{k}:{test_topo_history[k][-1]:.4f}" for k in sorted(topo_count)]
        print(f"Epoch {ep+1}/{config['training']['n_epochs']}, Train L2: {train_loss:.6f}, Test L2: {test_loss:.6f}, TestByTopology[{', '.join(topo_parts)}], Time: {default_timer()-t1:.2f}s")

        if wandb_run is not None:
            metrics = {"epoch": ep + 1, "train/l2": train_loss, "test/l2": test_loss}
            for topo in sorted(topo_count):
                metrics[f"test/topology_{topo}"] = test_topo_history[topo][-1]
            wandb_run.log(metrics, step=ep + 1)

    out_dir = config.get("output", {}).get("dir", "results")
    prefix = os.path.join(out_dir, "thingi10k_topos")
    save_loss_plots(train_losses, test_losses, test_topo_history, prefix)
    print(f"[*] Saved Thingi10K TOPOS loss plots: {prefix}_overall.png and {prefix}_per_topology.png")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
