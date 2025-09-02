import torch
import torch.nn.functional as F

from typing import Optional
from torch.utils.checkpoint import checkpoint

from .relative_grid_function import RelativeGridAttnCUDAFunction, relative_grid_attn_python, residual_conv_block


class RelativeGridAttention(torch.nn.Module):
    """
    nn.Module wrapper for the fused_attention operation.
    """
    def __init__(
        self, 
        dim,
        res,
        hidden_dim=1,
        kernel_size=3,
        implementation: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.res = res if isinstance(res,tuple) else (res,res)
        assert implementation in ["python", "cuda"]
        self.implementation = implementation

        rel_bias = torch.randn(self.res[0], self.res[1], dim)
        self.rel_bias = torch.nn.Parameter(rel_bias)
        
        conv1_weight = torch.randn(hidden_dim, 1, kernel_size, kernel_size)
        self.conv1_weight = torch.nn.Parameter(conv1_weight)

        conv1_bias = torch.randn(hidden_dim)
        self.conv1_bias = torch.nn.Parameter(conv1_bias)

        conv2_weight = torch.randn(1,hidden_dim, kernel_size, kernel_size)
        self.conv2_weight = torch.nn.Parameter(conv2_weight)

        conv2_bias = torch.randn(1)
        self.conv2_bias = torch.nn.Parameter(conv2_bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        pos: torch.Tensor,
        implementation: Optional[str] = None
    ) -> torch.Tensor:
        B, Q, C = queries.shape
        _, H, W, _ = keys.shape

        implementation_to_use = self.implementation if implementation is None else implementation

        if implementation_to_use == "cuda":
            output = RelativeGridAttnCUDAFunction.apply(
                queries, keys, pos, self.rel_bias
            )
            # Add residual convolution + ReLU + convolution
            # Reshape output to add channel dimension: [B, Q, H, W] -> [B*Q, 1, H, W]
            output_reshaped = output.view(B*Q, 1, H, W)
            
            # Apply checkpointed residual block
            # The ReLU activation inside this block won't be stored in memory
            x = checkpoint(
                residual_conv_block,
                output_reshaped,
                self.conv1_weight,
                self.conv1_bias,
                self.conv2_weight,
                self.conv2_bias,
                use_reentrant=False
            )
            
            # Residual connection
            output_reshaped = output_reshaped + x
            
            # Reshape back to original dimensions: [B*Q, 1, H, W] -> [B, Q, H, W]
            output = output_reshaped.view(B, Q, H, W)

        elif implementation_to_use == "python":
            output = relative_grid_attn_python(
                queries, keys, pos, self.rel_bias,
                self.conv1_weight, self.conv1_bias,
                self.conv2_weight, self.conv2_bias
            )
        
        return output