import torch

from ..functions.relative_grid_function import RelativeGridAttnCUDAFunction, relative_grid_attn_python


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
    ) -> torch.Tensor:
        if self.implementation == "cuda":
            return RelativeGridAttnCUDAFunction.apply(
                queries, keys, pos, rel_bias
            )
        elif self.implementation == "python":
            return relative_grid_attn_python(
                queries, keys, pos, rel_bias
            )