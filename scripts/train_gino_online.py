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
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neuralop.models import GINO
from topos.data import SyntheticGeometryDatasetGINO
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


def save_loss_plots(train_losses, test_losses, test_topology_history, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train L2")
    plt.plot(epochs, test_losses, label="Test L2")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GINO Online Loss Curves")
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
    plt.title("GINO Test Loss by Source Topology")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_per_topology.png", dpi=160)
    plt.close()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="GINO Online Trainer (Mixed-Geometry SDF Baseline)")
    parser.add_argument("--config", type=str, default="configs/mixed_genus_fair_comparison.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="otno-topos-mixed-geometry")
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", "mamtapc003-zenteiq-ai"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="auto", help='GPU selection: "auto", "cpu", "0", "1", "0,1", or "all"')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    set_seed(config["training"].get("seed", 42))
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
                    tags=["gino", "official", "online", "mixed-genus", "sdf"],
                )
            except Exception as e:
                print(f"[!] wandb init failed: {e}. Continuing without wandb logging.")
                wandb_run = None

    gino_config = config.get("gino_official", {})
    latent_grid_size = tuple(gino_config.get("latent_grid_size", [16, 16, 16]))
    cache_dir = config["dataset"].get("cache_dir", None)
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    train_ds = SyntheticGeometryDatasetGINO(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="train",
        expand_factor=config["dataset"].get("expand_factor", 2.0),
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
        latent_grid_size=latent_grid_size,
    )
    test_ds = SyntheticGeometryDatasetGINO(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"].get("expand_factor", 2.0),
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
        latent_grid_size=latent_grid_size,
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    pressures = []
    features = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        pressures.append(batch["pressure"])
        features.append(batch["features"])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, dim=0), reduce_dim=[0, 1]).to(device)
    feat_norm = UnitGaussianNormalizer(torch.cat(features, dim=0), reduce_dim=[0, 1]).to(device)

    gino_in_channels = gino_config.get("in_channels", 1)
    n_modes = tuple(gino_config.get("n_modes", [16, 16, 16]))
    hidden_channels = gino_config.get("hidden_channels", config["model"]["hidden_channels"])
    model = GINO(
        in_channels=gino_in_channels,
        out_channels=config["model"]["out_channels"],
        latent_feature_channels=gino_config.get("latent_feature_channels", None),
        gno_coord_dim=3,
        in_gno_radius=gino_config.get("in_gno_radius", 0.2),
        out_gno_radius=gino_config.get("out_gno_radius", 0.2),
        in_gno_transform_type=gino_config.get("in_gno_transform_type", "linear"),
        out_gno_transform_type=gino_config.get("out_gno_transform_type", "linear"),
        in_gno_pos_embed_type=gino_config.get("in_gno_pos_embed_type", "transformer"),
        out_gno_pos_embed_type=gino_config.get("out_gno_pos_embed_type", "transformer"),
        fno_in_channels=gino_in_channels,
        fno_n_modes=n_modes,
        fno_hidden_channels=hidden_channels,
        fno_n_layers=gino_config.get("n_layers", config["model"]["n_layers"]),
        gno_embed_channels=gino_config.get("gno_embed_channels", 32),
        in_gno_channel_mlp_hidden_layers=gino_config.get("in_gno_channel_mlp_hidden_layers", [80, 80, 80]),
        out_gno_channel_mlp_hidden_layers=gino_config.get("out_gno_channel_mlp_hidden_layers", [256, 256]),
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
        fno_factorization=gino_config.get("factorization", config["model"].get("factorization", None)),
        fno_rank=gino_config.get("rank", config["model"].get("rank", 1.0)),
        fno_norm=gino_config.get("norm", config["model"].get("norm", None)),
        fno_use_channel_mlp=gino_config.get("use_mlp", config["model"].get("use_mlp", True)),
        fno_channel_mlp_expansion=gino_config.get("mlp_expansion", config["model"].get("mlp", {}).get("expansion", 0.5)),
        fno_channel_mlp_dropout=gino_config.get("mlp_dropout", config["model"].get("mlp", {}).get("dropout", 0.0)),
    )
    model, device, gpu_ids = prepare_model_for_devices(model, args.gpus)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["step_size"], gamma=config["training"]["gamma"])
    loss_fn = LpLoss(size_average=False)

    train_losses, test_losses = [], []
    test_topo_history = defaultdict(list)
    epochs = config["training"]["n_epochs"]
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x = feat_norm.encode(batch["features"].to(device).clone())
            y = pressure_norm.encode(batch["pressure"].to(device).clone()).unsqueeze(-1)
            out = model(
                input_geom=batch["input_geom"].to(device),
                latent_queries=batch["latent_queries"].to(device),
                output_queries=batch["output_queries"].to(device),
                x=x,
            )
            loss = loss_fn(out.squeeze(-1), y.squeeze(-1))
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
                out = model(
                    input_geom=batch["input_geom"].to(device),
                    latent_queries=batch["latent_queries"].to(device),
                    output_queries=batch["output_queries"].to(device),
                    x=feat_norm.encode(batch["features"].to(device).clone()),
                )
                out = pressure_norm.decode(out.squeeze(-1).clone())
                y = batch["pressure"].to(device)
                loss_value = loss_fn(out, y).item()
                test_loss += loss_value
                for bi, topo in enumerate(batch["source_topology"]):
                    lv = loss_fn(out[bi:bi + 1], y[bi:bi + 1]).item()
                    topo_loss[topo] += lv
                    topo_count[topo] += 1

        train_loss /= len(train_ds)
        test_loss /= len(test_ds)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        for topo in sorted(topo_count.keys()):
            test_topo_history[topo].append(topo_loss[topo] / max(topo_count[topo], 1))
        topo_parts = [f"{k}:{test_topo_history[k][-1]:.4f}" for k in sorted(topo_count.keys())]
        print(f"Epoch {ep+1}/{epochs}, Train L2: {train_loss:.6f}, Test L2: {test_loss:.6f}, TestByTopology[{', '.join(topo_parts)}], Time: {default_timer()-t1:.2f}s")

        if wandb_run is not None:
            metrics = {
                "epoch": ep + 1,
                "train/l2": train_loss,
                "test/l2": test_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            for topo in sorted(topo_count.keys()):
                metrics[f"test/topology_{topo}"] = test_topo_history[topo][-1]
            wandb_run.log(metrics, step=ep + 1)

    out_dir = config.get("output", {}).get("dir", "results")

    chk_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(chk_dir, exist_ok=True)
    chk_path = os.path.join(chk_dir, "gino_mixed_genus.pt")
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(model_state, chk_path)
    print(f"[*] Saved GINO weights to: {chk_path}")

    prefix = os.path.join(out_dir, "gino_online_mixed")
    save_loss_plots(train_losses, test_losses, test_topo_history, prefix)
    print(f"[*] Saved GINO loss plots: {prefix}_overall.png and {prefix}_per_topology.png")
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
