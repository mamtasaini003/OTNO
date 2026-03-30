import torch
import torch.nn as nn
from neuralop.layers.gno_block import GNOBlock

class GraphFallbackSolver(nn.Module):
    """
    Graph Neural Operator Fallback Solver for topologies where Optimal Transport fails.
    This strictly operates on the unstructured physical coordinates via Message Passing 
    rather than mapping to a defined grid.
    
    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output prediction channels.
    hidden_channels : int
        Hidden representation size.
    radius : float
        Radius for Open3D neighbor search.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, radius=0.1):
        super().__init__()
        
        # Lift raw features to hidden dimension natively
        self.lifting = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Core Graph Kernel Integral operator
        self.gno = GNOBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            coord_dim=3,
            radius=radius,
            transform_type='nonlinear'
        )
        
        # Project hidden back to target continuous PDE space natively
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, points, features):
        """
        Forward pass entirely bypassing Transport/Latent Grids.
        
        Parameters
        ----------
        points : torch.Tensor of shape (B, N, 3)
            Physical space unstructured 3D node coordinates.
        features : torch.Tensor of shape (B, N, C)
            Input features at each physical node.
            
        Returns
        -------
        out : torch.Tensor of shape (B, N, out_channels)
            Continuous prediction natively in 3D graph space.
        """
        # GNO normally does not take batch dimensions effortlessly if graphs have varying sizes,
        # but since we strictly use constant N=3600 points per batch generally, we loop over the batch.
        B, N, _ = points.shape
        out_batch = []
        for b in range(B):
            # Lift features
            f_y = self.lifting(features[b]) # [N, hidden]
            
            # GNO applies exactly matching y=domain to x=queries
            h = self.gno(y=points[b], x=points[b], f_y=f_y) # [N, hidden]
            
            # Project down
            o = self.projection(h) # [N, out]
            out_batch.append(o)
            
        return torch.stack(out_batch, dim=0)

