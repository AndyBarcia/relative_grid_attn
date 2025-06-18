import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

try:
    import relative_grid_attn
except ImportError:
    print("CUDA extension fused_attn_ext not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


def residual_conv_block(output_reshaped, conv1_weight, conv1_bias, conv2_weight, conv2_bias):
    """
    Residual convolution block that will be checkpointed.
    This function encapsulates the operations where we don't want to store
    intermediate activations (specifically the ReLU output).
    """

    # Apply first convolution
    x = F.conv2d(
        output_reshaped,
        weight=conv1_weight,
        bias=conv1_bias,
        padding=1
    )
    # ReLU activation (this won't be stored in memory during forward pass)
    x = F.relu(x)
    # Apply second convolution
    x = F.conv2d(
        x,
        weight=conv2_weight,
        bias=conv2_bias,
        padding=1
    )
    return x


class RelativeGridAttnCUDAFunction(Function):
    @staticmethod
    def forward(ctx, queries, keys, pos, rel_bias):
        ctx.save_for_backward(queries, keys, pos, rel_bias)
        output = relative_grid_attn.forward(
            queries, keys, pos, rel_bias
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        queries, keys, pos, rel_bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_queries, grad_keys, grad_rel_bias = relative_grid_attn.backward(
            grad_output, queries, keys, pos, rel_bias
        )
        return grad_queries, grad_keys, None, grad_rel_bias, None, None


def relative_grid_attn_python(
    queries, 
    keys, 
    pos, 
    rel_bias,
    conv1_weight,  # [C_out=C, C_in=1, 3, 3]
    conv1_bias,    # [C_out=C]
    conv2_weight,  # [C_out=1, C_in=C, 3, 3]
    conv2_bias     # [C_out=1]
):
    """
    PyTorch reference implementation for the fused attention.
    queries:   [B, Q, C]
    keys:      [B, H, W, C]
    pos:       [B, Q, 4] (x_center, y_center, width, height) - normalized [0,1] or pixel scale
    rel_bias:  [H_rel, W_rel, C]
    conv1_weight: [1, 1, 3, 3] (output channels, input channels, height, width)
    conv1_bias: [1]
    conv2_weight: [1, 1, 3, 3]
    conv2_bias: [1]
    output:    [B, Q, H, W]
    """
    B, Q, C = queries.shape
    _B_keys, H, W, _C_keys = keys.shape
    _H_rel, _W_rel, _C_rel = rel_bias.shape
    C_inner = conv1_weight.shape[0]

    assert B == _B_keys, "Batch size mismatch"
    assert C == _C_keys, "Channel mismatch keys"
    assert C == _C_rel, "Channel mismatch rel_bias"
    assert conv1_weight.shape == (C_inner, 1, 3, 3), f"conv1_weight shape mismatch, got {conv1_weight.shape}, expected {(C_inner, 1, 3, 3)}"
    assert conv1_bias.shape == (C_inner,), f"conv1_bias shape mismatch, got {conv1_bias.shape}, expected {(C_inner,)}"
    assert conv2_weight.shape == (1, C_inner, 3, 3), f"conv2_weight shape mismatch, got {conv2_weight.shape}, expected {(1, C_inner, 3, 3)}"
    assert conv2_bias.shape == (1,), f"conv2_bias shape mismatch, got {conv2_bias.shape}, expected {(1,)}"

    scale = 1.0 / math.sqrt(C)

    # Content-based attention
    content_attn = einsum(queries, keys, "b q c, b h w c -> b q h w") * scale # (B, Q, H, W)

    # Relative position bias
    pos_x_center = pos[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    pos_y_center = pos[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    pos_width    = pos[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    pos_height   = pos[:, :, 3].unsqueeze(-1).unsqueeze(-1)

    grid_y_coords = torch.linspace(0, 1, H, device=queries.device, dtype=queries.dtype)
    grid_x_coords = torch.linspace(0, 1, W, device=queries.device, dtype=queries.dtype)
    grid_x, grid_y = torch.meshgrid(grid_x_coords, grid_y_coords, indexing='xy')

    grid_x_expanded = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y_expanded = grid_y.unsqueeze(0).unsqueeze(0)

    rel_coords_x = (pos_x_center - grid_x_expanded) / pos_width
    rel_coords_y = (pos_y_center - grid_y_expanded) / pos_height

    sampling_grid = torch.stack((rel_coords_x, rel_coords_y), dim=-1)

    rel_bias_gs_input = rearrange(rel_bias, "hr wr c -> 1 c hr wr").to(queries.dtype)
    rel_bias_gs_input_expanded = rel_bias_gs_input.expand(B * Q, C, _H_rel, _W_rel)

    sampling_grid_reshaped = rearrange(sampling_grid, "b q h w xy -> (b q) h w xy")

    sampled_rel_bias_C_first = F.grid_sample(
        rel_bias_gs_input_expanded,
        sampling_grid_reshaped,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    sampled_rel_bias = rearrange(sampled_rel_bias_C_first, "(b q) c h w -> b q h w c", b=B)
    rel_attn = einsum(queries, sampled_rel_bias, "b q c, b q h w c -> b q h w")

    output = content_attn + rel_attn # (B,Q,H,W)

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

    return output