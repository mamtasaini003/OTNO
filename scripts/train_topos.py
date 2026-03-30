import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
from timeit import default_timer
import numpy as np

# Structural reference from neuraloperator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topos.models import TOPOS
from topos.utils import LpLoss, UnitGaussianNormalizer, DictDataset
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TOPOS Unified Trainer")
    parser.add_argument('--config', type=str, default='configs/topos.yaml', help="Path to config file")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================")
    print(f"Starting TOPOS Training on {config['dataset']['name']}")
    print(f"Device: {device}")
    print(f"Total Epochs: {config['training']['n_epochs']}")
    print(f"==========================================")

    # 1. Model Initialization using factory/module
    model = TOPOS(
        spherical_config=config['model']['spherical_config'],
        volumetric_config=config['model'].get('volumetric_config'),
        toroidal_config=config['model'].get('toroidal_config')
    ).to(device)
    
    # 2. Dataset Simulation / Loading (Paths from config)
    # Note: In real setup, replace with your actual DataLoader
    print(f"[*] Loading dataset from {config['dataset']['path']}...")
    # Mocking data structure for demonstration
    n_train = config['dataset']['train_samples']
    n_test = config['dataset']['test_samples']
    
    # Standard Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['training']['learning_rate'], 
                                 weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config['training']['step_size'], 
                                                gamma=config['training']['gamma'])
    myloss = LpLoss(d=2, size_average=False)

    # Training Loop
    t_start = default_timer()
    for ep in range(config['training']['n_epochs']):
        t1 = default_timer()
        model.train()
        
        # Simplified batch loop for demonstration
        # In practice: for batch in train_loader: ...
        train_l2 = 0.0 # Dummy tracker
        
        # Simulated batch pass (Forward Stage 1-4)
        # B=1, C=9, H=W=32 (Latent Grid as shown in docs)
        grid_width = 32
        dummy_input = torch.randn(1, 9, grid_width, grid_width).to(device)
        dummy_idx = torch.arange(1000).to(device) # Latent Decoder Indices
        dummy_target = torch.randn(1000, 1).to(device)
        
        optimizer.zero_grad()
        # forward(transports, idx_decoder, topology="auto", chi=2.0)
        predict = model(transports=dummy_input, idx_decoder=dummy_idx, chi=2.0)
        
        loss = myloss(predict.view(1, -1), dummy_target.view(1, -1))
        loss.backward()
        optimizer.step()
        train_l2 = loss.item()

        scheduler.step()

        t2 = default_timer()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{config['training']['n_epochs']}, Time: {t2-t1:.4f}s, Train Loss: {train_l2:.6f}")

    total_time = default_timer() - t_start
    print(f"\n[SUCCESS] Training completed in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()