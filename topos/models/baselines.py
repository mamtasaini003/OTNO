import torch
import torch.nn as nn
from neuralop.models import FNO, GINO

class DeepONet(nn.Module):
    """Deep Operator Network (DeepONet) baseline."""
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        # branch_input: (B, branch_dim)
        # trunk_input: (B, N, trunk_dim)
        branch_out = self.branch(branch_input) # (B, out_dim)
        trunk_out = self.trunk(trunk_input)   # (B, N, out_dim)
        
        # Dot product across the out_dim
        out = torch.einsum("bd,bnd->bn", branch_out, trunk_out)
        return out + self.bias

class UFNO(nn.Module):
    """U-Net FNO baseline (multi-scale FNO)."""
    def __init__(self, in_channels, out_channels, n_modes=(16, 16), hidden_channels=32):
        super().__init__()
        # Simplified U-FNO using neuralop as core
        self.fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, 
                       in_channels=in_channels, out_channels=out_channels,
                       use_channel_mlp=True)

    def forward(self, x):
        return self.fno(x)

def model_factory(model_type, config):
    if model_type == "fno":
        return FNO(**config)
    elif model_type == "gino":
        return GINO(**config)
    elif model_type == "deeponet":
        return DeepONet(**config)
    elif model_type == "ufno":
        return UFNO(**config)
    elif model_type == "topos":
        from .topos import TOPOS
        return TOPOS(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
