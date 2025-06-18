import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .relative_grid_function import RelativeGridAttnCUDAFunction, relative_grid_attn_python, residual_conv_block


class RelativeGridAttention(torch.nn.Module):
    """
    nn.Module wrapper for the fused_attention operation.
    """
    def __init__(self, implementation: str = "cuda"):
        super().__init__()
        self.implementation = implementation

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        pos: torch.Tensor,
        rel_bias: torch.Tensor,
        conv1_weight,  # [C_out=C, C_in=1, 3, 3]
        conv1_bias,    # [C_out=C]
        conv2_weight,  # [C_out=1, C_in=C, 3, 3]
        conv2_bias     # [C_out=1]
    ) -> torch.Tensor:
        B, Q, C = queries.shape
        _, H, W, _ = keys.shape

        if self.implementation == "cuda":
            output = RelativeGridAttnCUDAFunction.apply(
                queries, keys, pos, rel_bias
            )
            # Add residual convolution + ReLU + convolution
            # Reshape output to add channel dimension: [B, Q, H, W] -> [B*Q, 1, H, W]
            output_reshaped = output.view(B*Q, 1, H, W)
            
            # Apply checkpointed residual block
            # The ReLU activation inside this block won't be stored in memory
            x = checkpoint(
                residual_conv_block,
                output_reshaped,
                conv1_weight,
                conv1_bias,
                conv2_weight,
                conv2_bias,
                use_reentrant=False
            )
            
            # Residual connection
            output_reshaped = output_reshaped + x
            
            # Reshape back to original dimensions: [B*Q, 1, H, W] -> [B, Q, H, W]
            output = output_reshaped.view(B, Q, H, W)

        elif self.implementation == "python":
            output = relative_grid_attn_python(
                queries, keys, pos, rel_bias
            )
        
        return output