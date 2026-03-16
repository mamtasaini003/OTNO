import os
import re

for file in ['train_navier_stokes.py', 'train_burgers.py', 'train_darcy.py']:
    with open(file, 'r') as f:
        content = f.read()
    
    # We replace from neuralop import get_model with TOPOS
    content = content.replace("from neuralop import H1Loss, LpLoss, Trainer, get_model",
                              "from neuralop import H1Loss, LpLoss, Trainer\nfrom topos.models.topos import TOPOS")
    
    # We replace model = get_model(config)
    replacement = """
    # TOPOS model setup for regular grids (Volumetric branch acts as 2D/3D regular FNO)
    volumetric_config = dict(
        n_modes=(config.model.n_modes[0], config.model.n_modes[1]),
        hidden_channels=config.model.hidden_channels,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        n_layers=config.model.n_layers,
    )
    model = TOPOS(
        spherical_config=dict(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4),
        volumetric_config=volumetric_config,
        toroidal_config=volumetric_config,
        chi_tol=0.5,
        default_topology='volumetric'
    )
    """
    content = re.sub(r'model = get_model\(config\)', replacement, content)
    
    with open(file, 'w') as f:
        f.write(content)

print("Refactored FNO bench scripts to use TOPOS.")
