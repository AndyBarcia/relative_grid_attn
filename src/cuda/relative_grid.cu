#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int THREADS_PER_BLOCK = 512;

// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t, typename index_t>
__device__ static inline scalar_t grid_sampler_unnormalize(
    scalar_t coord, 
    index_t size
) {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
}

template <typename index_t>
__device__ static inline bool within_bounds_2d(index_t h, index_t w, index_t H, index_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

__global__ void fused_attn_forward_kernel(
    const float* __restrict__ queries,   // [B, Q, C]
    const float* __restrict__ pos,       // [B, Q, 4]
    const float* __restrict__ rel_bias,  // [H_rel, W_rel, C]
    float* __restrict__ output,          // [B, Q, H, W]
    const int B,
    const int Q,
    const int C,
    const int H,
    const int W,
    const int H_rel,
    const int W_rel
) {
    const int total_elements = B * Q * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    // Decompose index into components
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int q = (idx / (H * W)) % Q;
    const int b = idx / (Q * H * W);

    const float* cur_query = &queries[b * Q * C + q * C];
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute relative coordinates
    float gx = (W > 1) ? static_cast<float>(w) / (W - 1) : 0.0f;
    float gy = (H > 1) ? static_cast<float>(h) / (H - 1) : 0.0f;

    float rel_x = (x_center - gx) / width;
    float rel_y = (y_center - gy) / height;

    float ix = grid_sampler_unnormalize(rel_x, W_rel);
    float iy = grid_sampler_unnormalize(rel_y, H_rel);

    int ix_nw = static_cast<int>(floorf(ix));
    int iy_nw = static_cast<int>(floorf(iy));
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    bool nw_in = within_bounds_2d(iy_nw, ix_nw, H_rel, W_rel);
    bool ne_in = within_bounds_2d(iy_ne, ix_ne, H_rel, W_rel);
    bool sw_in = within_bounds_2d(iy_sw, ix_sw, H_rel, W_rel);
    bool se_in = within_bounds_2d(iy_se, ix_se, H_rel, W_rel);

    if (!(nw_in || ne_in || sw_in || se_in)) {
        // If completely out of bounds (something quite common
        // given bboxes are small), avoid inner loop alltogether.
        return;
    }

    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    // Fused loop for content and relative attention
    float rel_val = 0.0f;
    for (int c = 0; c < C; c++) {
        // Load once per channel
        const float q_val = cur_query[c];
                
        // Relative attention
        float bias_val = 0.0f;
        if (nw_in) bias_val += rel_bias[iy_nw * (W_rel * C) + ix_nw * C + c] * nw;
        if (ne_in) bias_val += rel_bias[iy_ne * (W_rel * C) + ix_ne * C + c] * ne;
        if (sw_in) bias_val += rel_bias[iy_sw * (W_rel * C) + ix_sw * C + c] * sw;
        if (se_in) bias_val += rel_bias[iy_se * (W_rel * C) + ix_se * C + c] * se;
        rel_val += q_val * bias_val;
    }
    output[idx] += rel_val;
}

__global__ void attn_grad_rel_bias_backward_kernel(
    const float* __restrict__ grad_out,   // [B, Q, H, W]
    const float* __restrict__ queries,    // [B, Q, C]
    const float* __restrict__ pos,        // [B, Q, 4]
    const float* __restrict__ rel_bias,   // [H_rel, W_rel, C]
    float* __restrict__ grad_rel_bias,    // [H_rel, W_rel, C]
    const int B,
    const int Q,
    const int C,
    const int H,
    const int W,
    const int H_rel,
    const int W_rel,
    const int grad_rel_bias_memory_span
) {
    const int total_elements = B * Q * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    // Decompose index
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int q = (idx / (H * W)) % Q;
    const int b = idx / (Q * H * W);

    const float dL = grad_out[idx];
    const int query_idx = b * Q * C + q * C;
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute relative coordinates
    float gx = (W > 1) ? static_cast<float>(w) / (W - 1) : 0.0f;
    float gy = (H > 1) ? static_cast<float>(h) / (H - 1) : 0.0f;

    float rel_x = (x_center - gx) / width;
    float rel_y = (y_center - gy) / height;

    float ix = grid_sampler_unnormalize(rel_x, W_rel);
    float iy = grid_sampler_unnormalize(rel_y, H_rel);

    int ix_nw = static_cast<int>(floorf(ix));
    int iy_nw = static_cast<int>(floorf(iy));
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    bool nw_in = within_bounds_2d(iy_nw, ix_nw, H_rel, W_rel);
    bool ne_in = within_bounds_2d(iy_ne, ix_ne, H_rel, W_rel);
    bool sw_in = within_bounds_2d(iy_sw, ix_sw, H_rel, W_rel);
    bool se_in = within_bounds_2d(iy_se, ix_se, H_rel, W_rel);

    if (!(nw_in || ne_in || sw_in || se_in)) {
        return;
    }

    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    // Fused loop for gradients
    for (int c = 0; c < C; c++) {
        const float q_val = queries[query_idx + c];
        const float dL_q_val = dL * q_val;

        // Relative gradients rel_bias
        if (nw_in) {
            at::native::fastAtomicAdd(grad_rel_bias, iy_nw * (W_rel * C) + ix_nw * C + c, grad_rel_bias_memory_span, nw * dL_q_val, true);
        }
        if (ne_in) {
            at::native::fastAtomicAdd(grad_rel_bias, iy_ne * (W_rel * C) + ix_ne * C + c, grad_rel_bias_memory_span, ne * dL_q_val, true);
        }
        if (sw_in) {
            at::native::fastAtomicAdd(grad_rel_bias, iy_sw * (W_rel * C) + ix_sw * C + c, grad_rel_bias_memory_span, sw * dL_q_val, true);
        }
        if (se_in) {
            at::native::fastAtomicAdd(grad_rel_bias, iy_se * (W_rel * C) + ix_se * C + c, grad_rel_bias_memory_span, se * dL_q_val, true);
        }        
    }
}

__global__ void attn_grad_queries_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ rel_bias,
    const float* __restrict__ pos,
    float* __restrict__ grad_queries,
    const int B,
    const int Q,
    const int C,
    const int H,
    const int W,
    const int H_rel,
    const int W_rel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Q * C) return;

    int c = idx % C;
    int q = (idx / C) % Q;
    int b = idx / (C * Q);

    const float* p = pos + (b * Q + q) * 4;
    float x_center = p[0];
    float y_center = p[1];
    float width = p[2];
    float height = p[3];

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float gx = (W > 1) ? static_cast<float>(w) / (W - 1) : 0.0f;
            float gy = (H > 1) ? static_cast<float>(h) / (H - 1) : 0.0f;

            float rel_x = (x_center - gx) / width;
            float rel_y = (y_center - gy) / height;

            float ix = grid_sampler_unnormalize(rel_x, W_rel);
            float iy = grid_sampler_unnormalize(rel_y, H_rel);

            int ix_nw = static_cast<int>(floorf(ix));
            int iy_nw = static_cast<int>(floorf(iy));
            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;
            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;
            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            float nw = (ix_se - ix) * (iy_se - iy);
            float ne = (ix - ix_sw) * (iy_sw - iy);
            float sw = (ix_ne - ix) * (iy - iy_ne);
            float se = (ix - ix_nw) * (iy - iy_nw);

            float bias_val = 0.0f;
            if (within_bounds_2d(iy_nw, ix_nw, H_rel, W_rel)) {
                bias_val += rel_bias[iy_nw * W_rel * C + ix_nw * C + c] * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, H_rel, W_rel)) {
                bias_val += rel_bias[iy_ne * W_rel * C + ix_ne * C + c] * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, H_rel, W_rel)) {
                bias_val += rel_bias[iy_sw * W_rel * C + ix_sw * C + c] * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, H_rel, W_rel)) {
                bias_val += rel_bias[iy_se * W_rel * C + ix_se * C + c] * se;
            }

            float dL = grad_out[((b * Q + q) * H + h) * W + w];
            sum += dL * bias_val;
        }
    }
    grad_queries[(b * Q + q) * C + c] += sum;
}

torch::Tensor fused_attn_forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& pos,
    const torch::Tensor& rel_bias
) {
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(pos);
    CHECK_INPUT(rel_bias);
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int H = keys.size(1);
    const int W = keys.size(2);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
        
    auto keys_flat = keys.view({B, H * W, C});
    auto output = (torch::matmul(queries, keys_flat.transpose(1, 2)) * (1.0f / std::sqrt(C))).view({B, Q, H, W});
    
    const int total_elements = B * Q * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    fused_attn_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        queries.data_ptr<float>(),
        pos.data_ptr<float>(),
        rel_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, Q, C, H, W, H_rel, W_rel
    );
    
    return output;
}

std::vector<torch::Tensor> fused_attn_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& pos,
    const torch::Tensor& rel_bias
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(pos);
    CHECK_INPUT(rel_bias);
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int H = keys.size(1);
    const int W = keys.size(2);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
        
    auto grad_queries = torch::matmul(grad_out.view({B, Q, H * W}), keys.view({B, H * W, C})) * (1.0f / std::sqrt(C));
    auto grad_keys = torch::matmul(grad_out.view({B, Q, H * W}).transpose(1, 2), queries).view({B, H, W, C}) * (1.0f / std::sqrt(C));
    auto grad_rel_bias = torch::zeros_like(rel_bias);
    
    {
        const int total_elements = B * Q * H * W;
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        attn_grad_rel_bias_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<float>(),
            queries.data_ptr<float>(),
            pos.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            grad_rel_bias.data_ptr<float>(),
            B, Q, C, H, W, H_rel, W_rel,
            static_cast<int>(grad_rel_bias.numel())
        );
    }

    {
        const int total_elements = B * Q * C;
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        attn_grad_queries_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            grad_out.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            pos.data_ptr<float>(),
            grad_queries.data_ptr<float>(),
            B, Q, C, H, W, H_rel, W_rel
        );
    }
    
    return {grad_queries, grad_keys, grad_rel_bias};
}

PYBIND11_MODULE(relative_grid_attn, m) {
    m.def("forward", &fused_attn_forward, "Fused Attention Forward");
    m.def("backward", &fused_attn_backward, "Fused Attention Backward");
}