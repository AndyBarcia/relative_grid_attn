#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

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

__global__ void relative_attention_forward_kernel(
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
    const int b = idx / (H * W * Q);

    const float* cur_query = &queries[b * Q * C + q * C];
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute normalized key coordinates
    const float gx = (W > 1) ? static_cast<float>(w) / (W - 1) : 0.0f;
    const float gy = (H > 1) ? static_cast<float>(h) / (H - 1) : 0.0f;
    
    // Compute relative coordinates
    float rel_x = (x_center - gx) / width;
    float rel_y = (y_center - gy) / height;
    
    // Map to grid coordinates
    float ix = grid_sampler_unnormalize(rel_x, W_rel);
    float iy = grid_sampler_unnormalize(rel_y, H_rel);
    
    int ix_nw = static_cast<int>(std::floor(ix));
    int iy_nw = static_cast<int>(std::floor(iy));

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    
    // get surfaces to each neighbor:
    float nw = (ix_se - ix)    * (iy_se - iy);
    float ne = (ix    - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix)    * (iy    - iy_ne);
    float se = (ix    - ix_nw) * (iy    - iy_nw);

    // Compute relative attention value
    float val = 0.0f;
    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        
        if (within_bounds_2d(iy_nw, ix_nw, H_rel, W_rel))
            bias_val += rel_bias[iy_nw * (W_rel*C) + ix_nw * C + c] * nw;
        if (within_bounds_2d(iy_ne, ix_ne, H_rel, W_rel))
            bias_val += rel_bias[iy_ne * (W_rel*C) + ix_ne * C + c] * ne;
        if (within_bounds_2d(iy_sw, ix_sw, H_rel, W_rel))
            bias_val += rel_bias[iy_sw * (W_rel*C) + ix_sw * C + c] * sw;
        if (within_bounds_2d(iy_se, ix_se, H_rel, W_rel))
            bias_val += rel_bias[iy_se * (W_rel*C) + ix_se * C + c] * se;

        val += cur_query[c] * bias_val;
    }
    output[idx] = val;
}

__global__ void relative_attention_backward_kernel(
    const float* __restrict__ grad_out,   // [B, Q, H, W]
    const float* __restrict__ queries,    // [B, Q, C]
    const float* __restrict__ pos,        // [B, Q, 4]
    const float* __restrict__ rel_bias,   // [H_rel, W_rel, C]
    float* grad_queries,     // [B, Q, C] (accumulate)
    float* grad_rel_bias,    // [H_rel, W_rel, C] (accumulate)
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

    // Decompose index
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int q = (idx / (H * W)) % Q;
    const int b = idx / (H * W * Q);

    const float dL = grad_out[idx];
    const float* cur_query = &queries[b * Q * C + q * C];
    const int query_idx = b * Q * C + q * C;
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute normalized key coordinates
    const float gx = (W > 1) ? static_cast<float>(w) / (W - 1) : 0.0f;
    const float gy = (H > 1) ? static_cast<float>(h) / (H - 1) : 0.0f;
    
    // Compute relative coordinates
    float rel_x = (x_center - gx) / width;
    float rel_y = (y_center - gy) / height;
    
    // Map to grid coordinates
    float ix = grid_sampler_unnormalize(rel_x, W_rel);
    float iy = grid_sampler_unnormalize(rel_y, H_rel);
    
    int ix_nw = static_cast<int>(std::floor(ix));
    int iy_nw = static_cast<int>(std::floor(iy));

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;
    
    // get surfaces to each neighbor:
    float nw = (ix_se - ix)    * (iy_se - iy);
    float ne = (ix    - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix)    * (iy    - iy_ne);
    float se = (ix    - ix_nw) * (iy    - iy_nw);

    // Compute gradients
    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        
        if (within_bounds_2d(iy_nw, ix_nw, H_rel, W_rel)) {
            bias_val += rel_bias[iy_nw * (W_rel*C) + ix_nw * C + c] * nw;
            atomicAdd(&grad_rel_bias[iy_nw * (W_rel*C) + ix_nw * C + c], dL * nw * cur_query[c]);
        }
        if (within_bounds_2d(iy_ne, ix_ne, H_rel, W_rel)) {
            bias_val += rel_bias[iy_ne * (W_rel*C) + ix_ne * C + c] * ne;
            atomicAdd(&grad_rel_bias[iy_ne * (W_rel*C) + ix_ne * C + c], dL * ne * cur_query[c]);
        }
        if (within_bounds_2d(iy_sw, ix_sw, H_rel, W_rel)) {
            bias_val += rel_bias[iy_sw * (W_rel*C) + ix_sw * C + c] * sw;
            atomicAdd(&grad_rel_bias[iy_sw * (W_rel*C) + ix_sw * C + c], dL * sw * cur_query[c]);
        }
        if (within_bounds_2d(iy_se, ix_se, H_rel, W_rel)) {
            bias_val += rel_bias[iy_se * (W_rel*C) + ix_se * C + c] * se;
            atomicAdd(&grad_rel_bias[iy_se * (W_rel*C) + ix_se * C + c], dL * se * cur_query[c]);
        }
        
        // Gradient for queries
        atomicAdd(&grad_queries[query_idx + c], dL * bias_val);
    }
}

torch::Tensor fused_attn_forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& pos,
    const torch::Tensor& rel_bias,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index
) {
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(pos);
    CHECK_INPUT(rel_bias);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int total_L = keys.size(1);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
    const int L = spatial_shapes.size(0);
    
    auto options = torch::TensorOptions()
        .dtype(queries.dtype())
        .device(queries.device());
    
    // Compute content attention
    const float scale = 1.0f / std::sqrt(C);
    auto output = torch::matmul(queries, keys.transpose(1, 2)) * scale;
    
    // Process each level
    for (int l = 0; l < L; l++) {
        int H = spatial_shapes[l][0].item<int>();
        int W = spatial_shapes[l][1].item<int>();
        int start_index = level_start_index[l].item<int>();
        int num_elements = H * W;
        
        // Skip if no elements in this level
        if (num_elements <= 0) continue;
        
        auto rel_attn_level = torch::zeros({B, Q, H, W}, options);
        const int total_elements = B * Q * H * W;
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        relative_attention_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            queries.data_ptr<float>(),
            pos.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            rel_attn_level.data_ptr<float>(),
            B, Q, C, H, W, H_rel, W_rel
        );
        
        // Flatten and add to output
        auto flat_attn = rel_attn_level.reshape({B, Q, num_elements});
        output.slice(2, start_index, start_index + num_elements) += flat_attn;
    }
    
    return output;
}

std::vector<torch::Tensor> fused_attn_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& pos,
    const torch::Tensor& rel_bias,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(pos);
    CHECK_INPUT(rel_bias);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int total_L = keys.size(1);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
    const int L = spatial_shapes.size(0);
    
    auto options = torch::TensorOptions()
        .dtype(queries.dtype())
        .device(queries.device());
    
    // Initialize gradients
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);
    auto grad_rel_bias = torch::zeros_like(rel_bias);
    
    // Compute content attention gradients
    const float scale = 1.0f / std::sqrt(C);
    grad_queries = torch::matmul(grad_output, keys) * scale;
    grad_keys = torch::matmul(grad_output.transpose(1, 2), queries) * scale;
    
    // Process each level for relative attention gradients
    for (int l = 0; l < L; l++) {
        int H = spatial_shapes[l][0].item<int>();
        int W = spatial_shapes[l][1].item<int>();
        int start_index = level_start_index[l].item<int>();
        int num_elements = H * W;
        
        // Skip if no elements in this level
        if (num_elements <= 0) continue;
        
        // Extract gradient for this level. Must be made contigous
        // because the kernel expects contiguous memory.
        auto grad_level = grad_output.slice(2, start_index, start_index + num_elements).reshape({B, Q, H, W}).contiguous();
        
        const int total_elements = B * Q * H * W;
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        relative_attention_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            grad_level.data_ptr<float>(),
            queries.data_ptr<float>(),
            pos.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            grad_queries.data_ptr<float>(),
            grad_rel_bias.data_ptr<float>(),
            B, Q, C, H, W, H_rel, W_rel
        );
    }
    
    return {grad_queries, grad_keys, grad_rel_bias};
}

PYBIND11_MODULE(relative_grid_attn, m) {
    m.def("forward", &fused_attn_forward, "Fused Attention Forward (Multi-Resolution)");
    m.def("backward", &fused_attn_backward, "Fused Attention Backward (Multi-Resolution)");
}