import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function

try:
    import relative_grid_attn
except ImportError:
    print("CUDA extension fused_attn_ext not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class RelativeGridAttnCUDAFunction(Function):
    @staticmethod
    def forward(ctx, queries, keys, pos, rel_bias, spatial_shapes, level_start_index):
        ctx.save_for_backward(queries, keys, pos, rel_bias, spatial_shapes, level_start_index)
        output = relative_grid_attn.forward(
            queries, keys, pos, rel_bias, spatial_shapes, level_start_index
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        queries, keys, pos, rel_bias, spatial_shapes, level_start_index = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_queries, grad_keys, grad_rel_bias = relative_grid_attn.backward(
            grad_output, queries, keys, pos, rel_bias, spatial_shapes, level_start_index
        )
        return grad_queries, grad_keys, None, grad_rel_bias, None, None


def relative_grid_attn_python(
    queries, 
    keys, 
    pos, 
    rel_bias,
    spatial_shapes,
    level_start_index
):
    """
    PyTorch reference implementation for the fused attention.
    queries:   [B, Q, C]
    keys:      [B, HWL, C]
    pos:       [B, Q, 4] (x_center, y_center, width, height) - normalized [0,1] or pixel scale
    rel_bias:  [H_rel, W_rel, C]
    spatial_shapes: [L, 2] - height and width for each level of the keys.
    level_start_index: [L] - start index for each level in the keys.
    output:    [B, Q, HWL]
    """
    B, Q, C = queries.shape
    _B_keys, HWL, _C_keys = keys.shape
    _H_rel, _W_rel, _C_rel = rel_bias.shape

    assert B == _B_keys, "Batch size mismatch"
    assert C == _C_keys, "Channel mismatch keys"
    assert C == _C_rel, "Channel mismatch rel_bias"

    # Content-based attention
    scale = 1.0 / math.sqrt(C)
    content_attn = einsum(queries, keys, "b q c, b hwl c -> b q hwl") * scale # (B, Q, HWL)

    pos_x_center = pos[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    pos_y_center = pos[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    pos_width    = pos[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    pos_height   = pos[:, :, 3].unsqueeze(-1).unsqueeze(-1)

    # Then combine the content and relative attention at each resolution.
    for shape,start_index in zip(spatial_shapes, level_start_index):
        # Resolution of this level.
        H, W = shape

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

        # Perform bilinear interpolation
        # padding_mode='zeros': CUDA kernel implicitly zeros out contributions from out-of-bound access
        # align_corners=False: Matches the CUDA kernel's indexing logic for interpolation.
        sampled_rel_bias_C_first = F.grid_sample(
            rel_bias_gs_input_expanded,
            sampling_grid_reshaped,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ) # (B,Q,C,H,W)

        # Compute the relative attention at this resolution (_H_rel, _W_rel)
        sampled_rel_bias = rearrange(sampled_rel_bias_C_first, "(b q) c h w -> b q h w c", b=B)

        # TODO optimize knowing that rel_attn is mostly zeros (it is only non-zero within the
        # bound of the bounding boxes defined by pos).
        rel_attn = einsum(queries, sampled_rel_bias, "b q c, b q h w c -> b q h w")

        # Add the relative attention to the content attention.
        content_attn[:, :, start_index:start_index + H * W] += rel_attn.flatten(-2,-1)

    return content_attn
