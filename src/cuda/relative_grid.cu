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

__global__ void relative_attention_forward_fused_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ pos,
    const float* __restrict__ rel_bias,
    float* __restrict__ output,
    const int B,
    const int Q,
    const int C,
    const int total_L,
    const int H_rel,
    const int W_rel,
    const int L,
    const int* __restrict__ spatial_shapes,
    const int* __restrict__ level_start_index,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int remaining = idx;
    int l = 0;
    int H = 0, W = 0;
    
    // Process levels 0 to L-2
    for (; l < L - 1; l++) {
        H = spatial_shapes[2 * l];
        W = spatial_shapes[2 * l + 1];
        int level_elements = B * Q * H * W;
        if (remaining < level_elements) break;
        remaining -= level_elements;
    }
    
    // Last level
    if (l < L) {
        H = spatial_shapes[2 * l];
        W = spatial_shapes[2 * l + 1];
    } else {
        return;  // Invalid level
    }

    int b = remaining / (Q * H * W);
    int q = (remaining / (H * W)) % Q;
    int h = (remaining / W) % H;
    int w = remaining % W;

    const float* cur_query = queries + b * Q * C + q * C;
    const float* cur_pos = pos + b * Q * 4 + q * 4;
    float x_center = cur_pos[0];
    float y_center = cur_pos[1];
    float width = cur_pos[2];
    float height = cur_pos[3];

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

    float val = 0.0f;
    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        if (nw_in) bias_val += rel_bias[iy_nw * (W_rel * C) + ix_nw * C + c] * nw;
        if (ne_in) bias_val += rel_bias[iy_ne * (W_rel * C) + ix_ne * C + c] * ne;
        if (sw_in) bias_val += rel_bias[iy_sw * (W_rel * C) + ix_sw * C + c] * sw;
        if (se_in) bias_val += rel_bias[iy_se * (W_rel * C) + ix_se * C + c] * se;
        val += cur_query[c] * bias_val;
    }

    int key_idx = level_start_index[l] + h * W + w;
    int out_idx = b * Q * total_L + q * total_L + key_idx;
    atomicAdd(&output[out_idx], val);
}

__global__ void relative_attention_backward_fused_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ queries,
    const float* __restrict__ pos,
    const float* __restrict__ rel_bias,
    float* __restrict__ grad_queries,
    float* __restrict__ grad_rel_bias,
    const int B,
    const int Q,
    const int C,
    const int total_L,
    const int H_rel,
    const int W_rel,
    const int L,
    const int* __restrict__ spatial_shapes,
    const int* __restrict__ level_start_index,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int remaining = idx;
    int l = 0;
    int H = 0, W = 0;
    
    // Process levels 0 to L-2
    for (; l < L - 1; l++) {
        H = spatial_shapes[2 * l];
        W = spatial_shapes[2 * l + 1];
        int level_elements = B * Q * H * W;
        if (remaining < level_elements) break;
        remaining -= level_elements;
    }
    
    // Last level
    if (l < L) {
        H = spatial_shapes[2 * l];
        W = spatial_shapes[2 * l + 1];
    } else {
        return;  // Invalid level
    }

    int b = remaining / (Q * H * W);
    int q = (remaining / (H * W)) % Q;
    int h = (remaining / W) % H;
    int w = remaining % W;

    const float* cur_query = queries + b * Q * C + q * C;
    const float* cur_pos = pos + b * Q * 4 + q * 4;
    float x_center = cur_pos[0];
    float y_center = cur_pos[1];
    float width = cur_pos[2];
    float height = cur_pos[3];

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

    int key_idx = level_start_index[l] + h * W + w;
    int out_idx = b * Q * total_L + q * total_L + key_idx;
    float dL = grad_out[out_idx];

    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        if (nw_in) {
            bias_val += rel_bias[iy_nw * (W_rel * C) + ix_nw * C + c] * nw;
            atomicAdd(&grad_rel_bias[iy_nw * (W_rel * C) + ix_nw * C + c], dL * nw * cur_query[c]);
        }
        if (ne_in) {
            bias_val += rel_bias[iy_ne * (W_rel * C) + ix_ne * C + c] * ne;
            atomicAdd(&grad_rel_bias[iy_ne * (W_rel * C) + ix_ne * C + c], dL * ne * cur_query[c]);
        }
        if (sw_in) {
            bias_val += rel_bias[iy_sw * (W_rel * C) + ix_sw * C + c] * sw;
            atomicAdd(&grad_rel_bias[iy_sw * (W_rel * C) + ix_sw * C + c], dL * sw * cur_query[c]);
        }
        if (se_in) {
            bias_val += rel_bias[iy_se * (W_rel * C) + ix_se * C + c] * se;
            atomicAdd(&grad_rel_bias[iy_se * (W_rel * C) + ix_se * C + c], dL * se * cur_query[c]);
        }
        atomicAdd(&grad_queries[b * Q * C + q * C + c], dL * bias_val);
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
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int total_L = keys.size(1);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
    const int L = spatial_shapes.size(0);
    
    auto output = torch::matmul(queries, keys.transpose(1, 2)) * (1.0f / std::sqrt(C));
    
    if (L == 0) {
        return output;
    }

    // Move to device and ensure contiguous
    auto spatial_shapes_d = spatial_shapes.to(torch::kInt).to(queries.device()).contiguous();
    auto level_start_index_d = level_start_index.to(torch::kInt).to(queries.device()).contiguous();
    
    // Compute total elements on CPU
    auto spatial_shapes_cpu = spatial_shapes.to(torch::kCPU).contiguous();
    auto shapes_a = spatial_shapes_cpu.accessor<int, 2>();
    int total_elements = 0;
    for (int l = 0; l < L; l++) {
        int H = shapes_a[l][0];
        int W = shapes_a[l][1];
        total_elements += B * Q * H * W;
    }

    if (total_elements > 0) {
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        relative_attention_forward_fused_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            queries.data_ptr<float>(),
            pos.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            B, Q, C, total_L, H_rel, W_rel, L,
            spatial_shapes_d.data_ptr<int>(),
            level_start_index_d.data_ptr<int>(),
            total_elements
        );
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
    
    const int B = queries.size(0);
    const int Q = queries.size(1);
    const int C = queries.size(2);
    const int total_L = keys.size(1);
    const int H_rel = rel_bias.size(0);
    const int W_rel = rel_bias.size(1);
    const int L = spatial_shapes.size(0);
    
    auto grad_queries = torch::matmul(grad_output, keys) * (1.0f / std::sqrt(C));
    auto grad_keys = torch::matmul(grad_output.transpose(1, 2), queries) * (1.0f / std::sqrt(C));
    auto grad_rel_bias = torch::zeros_like(rel_bias);

    if (L == 0) {
        return {grad_queries, grad_keys, grad_rel_bias};
    }

    // Move to device and ensure contiguous
    auto spatial_shapes_d = spatial_shapes.to(torch::kInt).to(queries.device()).contiguous();
    auto level_start_index_d = level_start_index.to(torch::kInt).to(queries.device()).contiguous();
    
    // Compute total elements on CPU
    auto spatial_shapes_cpu = spatial_shapes.to(torch::kCPU).contiguous();
    auto shapes_a = spatial_shapes_cpu.accessor<int, 2>();
    int total_elements = 0;
    for (int l = 0; l < L; l++) {
        int H = shapes_a[l][0];
        int W = shapes_a[l][1];
        total_elements += B * Q * H * W;
    }

    if (total_elements > 0) {
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        relative_attention_backward_fused_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            grad_output.contiguous().data_ptr<float>(),
            queries.contiguous().data_ptr<float>(),
            pos.contiguous().data_ptr<float>(),
            rel_bias.contiguous().data_ptr<float>(),
            grad_queries.data_ptr<float>(),
            grad_rel_bias.data_ptr<float>(),
            B, Q, C, total_L, H_rel, W_rel, L,
            spatial_shapes_d.data_ptr<int>(),
            level_start_index_d.data_ptr<int>(),
            total_elements
        );
    }
    
    return {grad_queries, grad_keys, grad_rel_bias};
}

PYBIND11_MODULE(relative_grid_attn, m) {
    m.def("forward", &fused_attn_forward, "Fused Attention Forward (Multi-Resolution)");
    m.def("backward", &fused_attn_backward, "Fused Attention Backward (Multi-Resolution)");
}