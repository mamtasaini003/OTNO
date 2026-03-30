"""
Spherical / Toroidal Transport FNO for TOPOS Architecture.

Refactored from the original shapenet/topos/TransportFNO.py into a reusable
module. The spherical (genus-0) case uses the Spherical FNO (SFNO) architecture,
while the toroidal (genus-1) case uses standard 2D spectral convolutions on periodic grids.
"""

import torch
import torch.nn as nn

from neuralop.models import FNO, SFNO
from neuralop.layers.channel_mlp import ChannelMLP as NeuralopMLP
from neuralop.layers.spectral_convolution import SpectralConv


class SphericalTransportFNO(SFNO):
    """Transport-based SFNO on a spherical latent space.

    Extends neuralop.SFNO with an OT-based decode step that maps the latent
    solution back to the original irregular mesh vertices via index lookup.
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
            **kwargs
    ):
        nn.Module.__init__(self)
        lifting_ratio = max(lifting_channels / hidden_channels, 1.0)
        projection_ratio = max(projection_channels / hidden_channels, 1.0)

        SFNO.__init__(
            self,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channel_ratio=lifting_ratio,
            projection_channel_ratio=projection_ratio,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            use_channel_mlp=use_mlp,
            channel_mlp_dropout=mlp['dropout'] if mlp else 0,
            channel_mlp_expansion=mlp['expansion'] if mlp else 0.5,
            non_linearity=non_linearity,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=mlp_skip,
            separable=separable,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            **kwargs
        )

        # Override the projection head to operate on 1D (point-cloud) data
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
        """Forward pass through the SFNO solver + OT decoder."""
        # Lifting
        transports = self.lifting(transports)

        # Domain padding
        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        # FNO blocks (spherical convolution layers)
        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        # Unpad
        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports)

        # Reshape from 2D grid to point cloud: (B, hidden_channels, H, W) -> (B, hidden_channels, N)
        B = transports.shape[0]
        transports = transports.reshape(B, self.hidden_channels, -1)

        # OT Decoder: select points corresponding to original mesh vertices
        out = transports[:, :, idx_decoder]  # (B, hidden_channels, n_target)

        # Project to output channels
        # NeuralopMLP(n_dim=1) expects (B, Ch, N)
        out = self.projection(out)
        return out


class ToroidalTransportFNO(FNO):
    """Transport-based FNO on a 2D periodic grid (torus workbench).

    Extends neuralop.FNO with an OT-based decode step that maps the latent
    solution back to the original irregular mesh vertices via index lookup.
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

        FNO.__init__(
            self,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channel_ratio=lifting_ratio,
            projection_channel_ratio=projection_ratio,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            use_channel_mlp=use_mlp,
            channel_mlp_dropout=mlp['dropout'] if mlp else 0,
            channel_mlp_expansion=mlp['expansion'] if mlp else 0.5,
            non_linearity=non_linearity,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=mlp_skip,
            separable=separable,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            conv_module=SpectralConvLayer,
            **kwargs
        )

        # Override the projection head to operate on 1D (point-cloud) data
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
        """Forward pass through the 2D spectral solver + OT decoder."""
        # Lifting
        transports = self.lifting(transports)

        # Domain padding
        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        # FNO blocks (spectral convolution layers)
        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        # Unpad
        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports)

        # Reshape from 2D grid to point cloud: (B, hidden_channels, H, W) -> (B, hidden_channels, N)
        B = transports.shape[0]
        transports = transports.reshape(B, self.hidden_channels, -1)

        # OT Decoder: select points corresponding to original mesh vertices
        out = transports[:, :, idx_decoder]  # (B, hidden_channels, n_target)

        # Project to output channels
        # NeuralopMLP(n_dim=1) expects (B, Ch, N)
        out = self.projection(out)
        return out
