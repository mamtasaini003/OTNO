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
from topos.models.baselines import DeepONet
from topos.data import SyntheticGeometryDatasetDeepONet
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
    plt.title("DeepONet Online Loss Curves")
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
    plt.title("DeepONet Test Loss by Source Topology")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_per_topology.png", dpi=160)
    plt.close()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DeepONet Online Trainer (Mixed-Geometry Point-Cloud Baseline)")
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
                    tags=["deeponet", "online", "mixed-genus", "point-cloud"],
                )
            except Exception as e:
                print(f"[!] wandb init failed: {e}. Continuing without wandb logging.")
                wandb_run = None

    cache_dir = config["dataset"].get("cache_dir", None)
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "synthetic")

    deeponet_config = config.get("deeponet", {})
    branch_points = deeponet_config.get("branch_points", 256)
    train_ds = SyntheticGeometryDatasetDeepONet(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="train",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
        branch_points=branch_points,
    )
    test_ds = SyntheticGeometryDatasetDeepONet(
        cache_dir=cache_dir,
        n_train=config["dataset"]["train_samples"],
        n_test=config["dataset"]["test_samples"],
        split="test",
        expand_factor=config["dataset"]["expand_factor"],
        num_points=config["dataset"].get("num_points", 3586),
        base_seed=config["training"].get("seed", 42),
        branch_points=branch_points,
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    pressures = []
    branches = []
    for i, b in enumerate(train_loader):
        if i >= 50:
            break
        pressures.append(b["pressure"])
        branches.append(b["branch_input"])
    pressure_norm = UnitGaussianNormalizer(torch.cat(pressures, dim=0), reduce_dim=[0, 1]).to(device)
    branch_norm = UnitGaussianNormalizer(torch.cat(branches, dim=0), reduce_dim=[0]).to(device)

    hidden = config["model"]["hidden_channels"]
    model = DeepONet(branch_dim=branch_points * 3, trunk_dim=3, hidden_dim=hidden, out_dim=hidden)
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
        tr = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            branch = branch_norm.encode(b["branch_input"].to(device).clone())
            trunk = b["trunk_input"].to(device)
            y = pressure_norm.encode(b["pressure"].to(device).clone())  # (B,N)
            out = model(branch, trunk)  # (B,N)
            loss = loss_fn(out, y)
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
                branch = branch_norm.encode(b["branch_input"].to(device).clone())
                trunk = b["trunk_input"].to(device)
                y = b["pressure"].to(device)
                topo = b["source_topology"][0]
                out = model(branch, trunk)
                out = pressure_norm.decode(out.clone())
                lv = loss_fn(out, y).item()
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
        print(f"Epoch {ep+1}/{epochs}, Train L2: {tr:.6f}, Test L2: {te:.6f}, TestByTopology[{', '.join(topo_parts)}], Time: {default_timer()-t1:.2f}s")
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
    prefix = os.path.join(out_dir, "deeponet_online_mixed")
    save_loss_plots(train_losses, test_losses, test_topo_history, prefix)
    print(f"[*] Saved DeepONet loss plots: {prefix}_overall.png and {prefix}_per_topology.png")
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
