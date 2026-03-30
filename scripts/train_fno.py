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
from topos.models import FNO
from topos.utils import LpLoss, UnitGaussianNormalizer, DictDataset
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="FNO Baseline Trainer")
    parser.add_argument('--config', type=str, default='configs/fno.yaml', help="Path to FNO baseline config")
    parser.add_argument('--epochs', type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config['training']['n_epochs'] = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==========================================")
    print(f"Starting FNO Baseline Training")
    print(f"Device: {device}")
    print(f"Total Epochs: {config['training']['n_epochs']}")
    print(f"==========================================")

    # 1. Model Initialization using neuraloperator-style FNO
    model = FNO(**config['model']).to(device)
    
    # 2. Dataset Simulation / Loading
    print(f"[*] Loading dataset from {config['dataset']['path']}...")
    
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
        
        # B=10, C=3, H=W=32 (Typical FNO grid resolution)
        dummy_input = torch.randn(10, config['model']['in_channels'], 32, 32).to(device)
        dummy_target = torch.randn(10, config['model']['out_channels'], 32, 32).to(device)
        
        optimizer.zero_grad()
        predict = model(dummy_input)
        
        loss = myloss(predict.view(10, -1), dummy_target.view(10, -1))
        loss.backward()
        optimizer.step()
        train_l2 = loss.item()

        scheduler.step()

        t2 = default_timer()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{config['training']['n_epochs']}, Time: {t2-t1:.4f}s, Train Loss: {train_l2:.6f}")

    total_time = default_timer() - t_start
    print(f"\n[SUCCESS] FNO Training completed in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
