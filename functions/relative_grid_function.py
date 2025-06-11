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
    def forward(ctx, queries, keys, pos, rel_bias, grid_x, grid_y):
        ctx.save_for_backward(queries, keys, pos, rel_bias, grid_x, grid_y)
        output = relative_grid_attn.forward(
            queries, keys, pos, rel_bias, grid_x, grid_y
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        queries, keys, pos, rel_bias, grid_x, grid_y = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_queries, grad_keys, grad_rel_bias = relative_grid_attn.backward(
            grad_output, queries, keys, pos, rel_bias, grid_x, grid_y
        )
        return grad_queries, grad_keys, None, grad_rel_bias, None, None


def relative_grid_attn_python(queries, keys, pos, rel_bias, grid_x, grid_y):
    """
    PyTorch reference implementation for the fused attention.
    queries:   [B, Q, C]
    keys:      [B, H, W, C]
    pos:       [B, Q, 4] (x_center, y_center, width, height) - normalized [0,1] or pixel scale
    rel_bias:  [H_rel, W_rel, C]
    grid_x:    [H, W] - normalized [0,1] or pixel scale, same as pos
    grid_y:    [H, W] - normalized [0,1] or pixel scale, same as pos
    output:    [B, Q, H, W]
    """
    B, Q, C = queries.shape
    _B_keys, H, W, _C_keys = keys.shape
    _H_rel, _W_rel, _C_rel = rel_bias.shape

    assert B == _B_keys, "Batch size mismatch"
    assert C == _C_keys, "Channel mismatch keys"
    assert C == _C_rel, "Channel mismatch rel_bias"

    scale = 1.0 / math.sqrt(C)

    # Content-based attention
    # queries: (B, Q, C) -> (B, Q, 1, 1, C)
    # keys:    (B, H, W, C) -> (B, 1, H, W, C)
    # content_attn = torch.sum(queries.unsqueeze(2).unsqueeze(2) * keys.unsqueeze(1), dim=-1) * scale
    # Using einsum is cleaner:
    content_attn = einsum(queries, keys, "b q c, b h w c -> b q h w") * scale # (B, Q, H, W)

    # Relative position bias
    # pos has (x_center, y_center, width, height) for each query
    # grid_x, grid_y are coordinates of key positions

    # Expand dims for broadcasting
    # pos_x_center, etc: (B, Q, 1, 1)
    pos_x_center = pos[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    pos_y_center = pos[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    pos_width    = pos[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    pos_height   = pos[:, :, 3].unsqueeze(-1).unsqueeze(-1)

    # grid_x_exp, grid_y_exp: (1, 1, H, W)
    grid_x_expanded = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y_expanded = grid_y.unsqueeze(0).unsqueeze(0)

    # Calculate relative coordinates `rel_x`, `rel_y` for grid_sample.
    # These should be in [-1, 1] range for grid_sample.
    # The CUDA kernel uses: rel_x = (x_center - gx) / width
    # If gx = x_center - width  => rel_x = 1
    # If gx = x_center + width  => rel_x = -1
    # This is the correct range for grid_sample if `align_corners=True` for the interpretation
    # of what "width" means.
    # However, the CUDA kernel's mapping:
    # u = (rel_x + 1.0f) * W_rel * 0.5f - 0.5f;
    # This formula is characteristic of `align_corners=False` in grid_sample,
    # where rel_x in [-1, 1] maps to indices for a grid of size W_rel.
    # So, the (pos_x_center - grid_x_expanded) / pos_width gives the correct [-1,1] normalized coords.

    rel_coords_x = (pos_x_center - grid_x_expanded) / pos_width
    rel_coords_y = (pos_y_center - grid_y_expanded) / pos_height

    # Stack to form the sampling grid for grid_sample: (B, Q, H, W, 2)
    # grid_sample expects (x, y) order for the last dimension.
    sampling_grid = torch.stack((rel_coords_x, rel_coords_y), dim=-1)

    # Reshape for grid_sample:
    # rel_bias needs to be (N, C_in, D_in, H_in, W_in) -> (1, C, H_rel, W_rel) for 2D
    # (using N=B*Q to process all at once)
    # rel_bias: (H_rel, W_rel, C) -> (1, C, H_rel, W_rel)
    rel_bias_gs_input = rearrange(rel_bias, "hr wr c -> 1 c hr wr").to(queries.dtype)
    # Expand for batch processing with grid_sample: (B*Q, C, H_rel, W_rel)
    rel_bias_gs_input_expanded = rel_bias_gs_input.expand(B * Q, C, _H_rel, _W_rel)

    # sampling_grid: (B, Q, H, W, 2) -> (B*Q, H, W, 2)
    sampling_grid_reshaped = rearrange(sampling_grid, "b q h w xy -> (b q) h w xy")

    # Perform bilinear interpolation
    # Output is (N, C, H_out, W_out) -> (B*Q, C, H, W)
    # padding_mode='zeros': CUDA kernel implicitly zeros out contributions from out-of-bound access
    # align_corners=False: Matches the CUDA kernel's indexing logic for interpolation.
    sampled_rel_bias_C_first = F.grid_sample(
        rel_bias_gs_input_expanded,
        sampling_grid_reshaped,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # Reshape sampled_rel_bias back: (B*Q, C, H, W) -> (B, Q, H, W, C)
    sampled_rel_bias = rearrange(sampled_rel_bias_C_first, "(b q) c h w -> b q h w c", b=B)

    # Dot product with queries
    # queries: (B, Q, C)
    # sampled_rel_bias: (B, Q, H, W, C)
    # rel_attn: (B, Q, H, W)
    rel_attn = einsum(queries, sampled_rel_bias, "b q c, b q h w c -> b q h w")

    output = content_attn + rel_attn
    return output
