"""
3D Volumetric Fourier Neural Operator for TOPOS Architecture.

Handles the "volumetric" branch for geometries that are best represented
in a regular Cartesian grid [0,1]³ — e.g., bounding-box regions, higher-genus
surfaces that don't map cleanly to sphere or torus.

Uses 3D spectral convolutions via neuralop's FNO with n_modes=(M1, M2, M3).
To compete with transformer-based local attention mechanisms (e.g., GAOT), 
ensure high enough mode-cut is used to capture high-frequency details 
in the [0,1]³ latent space.
"""

import torch
import torch.nn as nn
import numpy as np

from neuralop.models import FNO
from neuralop.layers.channel_mlp import ChannelMLP as NeuralopMLP
from neuralop.layers.spectral_convolution import SpectralConv


def create_cartesian_grid(n_x, n_y, n_z):
    """Create a uniform Cartesian grid on [0, 1]³.

    Parameters
    ----------
    n_x, n_y, n_z : int
        Number of grid points along each axis.

    Returns
    -------
    Tensor
        Shape (3, n_x, n_y, n_z) — coordinate channels.
    """
    x = torch.linspace(0, 1, n_x)
    y = torch.linspace(0, 1, n_y)
    z = torch.linspace(0, 1, n_z)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing='ij')
    return torch.stack([gx, gy, gz], dim=0)  # (3, n_x, n_y, n_z)


class VolumetricFNO(FNO):
    """3D Volumetric FNO for Cartesian latent domains.

    Extends neuralop.FNO to operate on 3D regular grids and includes an
    OT-based decoder step (index lookup) to map solutions back to the
    original irregular mesh vertices.

    Parameters
    ----------
    n_modes : tuple of int
        Number of Fourier modes per dimension, e.g., (16, 16, 16).
    hidden_channels : int
        Width of hidden layers.
    in_channels : int
        Number of input channels (transported coords + features).
    out_channels : int
        Number of output solution channels.
    projection_channels : int
        Width of the output projection MLP.
    **kwargs
        Forwarded to neuralop.FNO.
    """

    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=4,
            out_channels=1,
            lifting_channels=256,
            projection_channels=256,
            n_layers=4,
            positional_embedding=None,
            use_mlp=False,
            mlp=None,
            non_linearity=torch.nn.functional.gelu,
            norm=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            domain_padding=None,
            domain_padding_mode="one-sided",
            fft_norm="forward",
            SpectralConvLayer=SpectralConv,
            **kwargs
    ):
        nn.Module.__init__(self)
        lifting_ratio = max(lifting_channels / hidden_channels, 1.0)
        projection_ratio = max(projection_channels / hidden_channels, 1.0)

        # Use explicitly provided args or filter from config
        fno_kwargs = {
            'n_modes': n_modes,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_channels': hidden_channels,
            'lifting_channel_ratio': lifting_ratio,
            'projection_channel_ratio': projection_ratio,
            'n_layers': n_layers,
            'positional_embedding': positional_embedding,
            'use_channel_mlp': use_mlp,
            'channel_mlp_dropout': mlp['dropout'] if mlp else 0,
            'channel_mlp_expansion': mlp['expansion'] if mlp else 0.5,
            'non_linearity': non_linearity,
            'norm': norm,
            'preactivation': preactivation,
            'fno_skip': fno_skip,
            'channel_mlp_skip': mlp_skip,
            'separable': separable,
            'factorization': factorization,
            'rank': rank,
            'fixed_rank_modes': fixed_rank_modes,
            'implementation': implementation,
            'decomposition_kwargs': decomposition_kwargs,
            'domain_padding': domain_padding,
            'conv_module': SpectralConvLayer,
        }
        
        # Clean up any None or extra values if necessary, then call init
        FNO.__init__(self, **fno_kwargs)

        # Override projection to 1D (point-cloud output after decoder indexing)
        self.projection = NeuralopMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            non_linearity=non_linearity,
            n_dim=1,
        )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def forward(self, transports, idx_decoder):
        """Forward pass through the 3D spectral solver + OT decoder.

        Parameters
        ----------
        transports : Tensor
            Shape (1, in_channels, Nx, Ny, Nz).
            OT-transported features on the 3D latent Cartesian grid.
        idx_decoder : LongTensor
            Shape (n_target,). Indices mapping 3D grid points (flattened)
            to original mesh vertices.

        Returns
        -------
        Tensor
            Shape (1, out_channels, n_target) — predicted field on the
            physical mesh.
        """
        # Lifting
        transports = self.lifting(transports)

        # Domain padding (3D)
        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        # FNO blocks with 3D spectral convolutions
        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        # Unpad
        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports)

        # Reshape from 3D grid to point cloud:
        # (B, hidden_channels, Nx, Ny, Nz) → (B, hidden_channels, N)
        B = transports.shape[0]
        transports = transports.reshape(B, self.hidden_channels, -1)

        # OT Decoder: select points corresponding to original mesh vertices
        # transports: (B, hidden_channels, N_latent)
        # idx_decoder: (n_target,)
        out = transports[:, :, idx_decoder]  # (B, hidden_channels, n_target)

        # Project to output channels
        # NeuralopMLP(n_dim=1) expects (B, Ch, N)
        out = self.projection(out)
        return out
