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
    float u = (rel_x + 1.0f) * W_rel * 0.5f - 0.5f;
    float v = (rel_y + 1.0f) * H_rel * 0.5f - 0.5f;
    
    const int j0 = floorf(u);
    const int j1 = j0 + 1;
    const int i0 = floorf(v);
    const int i1 = i0 + 1;
    
    const float w00 = (i1 - v) * (j1 - u); 
    const float w01 = (i1 - v) * (u - j0);
    const float w10 = (v - i0) * (j1 - u);
    const float w11 = (v - i0) * (u - j0);
    
    // Check boundaries
    const bool in00 = (i0 >= 0) && (i0 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in01 = (i0 >= 0) && (i0 < H_rel) && (j1 >= 0) && (j1 < W_rel);
    const bool in10 = (i1 >= 0) && (i1 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in11 = (i1 >= 0) && (i1 < H_rel) && (j1 >= 0) && (j1 < W_rel);

    // Precompute rel_bias offsets
    const int offset00 = in00 ? (i0 * W_rel + j0) * C : -1;
    const int offset01 = in01 ? (i0 * W_rel + j1) * C : -1;
    const int offset10 = in10 ? (i1 * W_rel + j0) * C : -1;
    const int offset11 = in11 ? (i1 * W_rel + j1) * C : -1;

    // Compute relative attention value
    float val = 0.0f;
    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        if (in00) bias_val += w00 * rel_bias[offset00 + c];
        if (in01) bias_val += w01 * rel_bias[offset01 + c];
        if (in10) bias_val += w10 * rel_bias[offset10 + c];
        if (in11) bias_val += w11 * rel_bias[offset11 + c];
        val += cur_query[c] * bias_val;
    }
    output[idx] = val;
}

__global__ void relative_attention_backward_kernel(
    const float* __restrict__ grad_out,   // [B, Q, H, W]
    const float* __restrict__ queries,    // [B, Q, C]
    const float* __restrict__ pos,        // [B, Q, 4]
    const float* __restrict__ rel_bias,   // [H_rel, W_rel, C]
    float* __restrict__ grad_queries,     // [B, Q, C] (accumulate)
    float* __restrict__ grad_rel_bias,    // [H_rel, W_rel, C] (accumulate)
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
    float u = (rel_x + 1.0f) * W_rel * 0.5f - 0.5f;
    float v = (rel_y + 1.0f) * H_rel * 0.5f - 0.5f;
    
    const int j0 = floorf(u);
    const int j1 = j0 + 1;
    const int i0 = floorf(v);
    const int i1 = i0 + 1;
    
    const float w00 = (i1 - v) * (j1 - u);
    const float w01 = (i1 - v) * (u - j0);
    const float w10 = (v - i0) * (j1 - u);
    const float w11 = (v - i0) * (u - j0);
    
    // Check boundaries
    const bool in00 = (i0 >= 0) && (i0 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in01 = (i0 >= 0) && (i0 < H_rel) && (j1 >= 0) && (j1 < W_rel);
    const bool in10 = (i1 >= 0) && (i1 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in11 = (i1 >= 0) && (i1 < H_rel) && (j1 >= 0) && (j1 < W_rel);

    // Precompute interpolation weights scaled by dL
    const float dL_w00 = dL * w00;
    const float dL_w01 = dL * w01;
    const float dL_w10 = dL * w10;
    const float dL_w11 = dL * w11;

    // Precompute rel_bias offsets
    const int offset00 = in00 ? (i0 * W_rel + j0) * C : -1;
    const int offset01 = in01 ? (i0 * W_rel + j1) * C : -1;
    const int offset10 = in10 ? (i1 * W_rel + j0) * C : -1;
    const int offset11 = in11 ? (i1 * W_rel + j1) * C : -1;

    // Compute gradients
    for (int c = 0; c < C; c++) {
        float bias_val = 0.0f;
        
        if (in00) {
            bias_val += w00 * rel_bias[offset00 + c];
            atomicAdd(&grad_rel_bias[offset00 + c], dL_w00 * cur_query[c]);
        }
        if (in01) {
            bias_val += w01 * rel_bias[offset01 + c];
            atomicAdd(&grad_rel_bias[offset01 + c], dL_w01 * cur_query[c]);
        }
        if (in10) {
            bias_val += w10 * rel_bias[offset10 + c];
            atomicAdd(&grad_rel_bias[offset10 + c], dL_w10 * cur_query[c]);
        }
        if (in11) {
            bias_val += w11 * rel_bias[offset11 + c];
            atomicAdd(&grad_rel_bias[offset11 + c], dL_w11 * cur_query[c]);
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
    auto content_attn = torch::matmul(queries, keys.transpose(1, 2)) * scale;
    auto output = content_attn.clone();
    
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
        
        // Extract gradient for this level
        auto grad_level = grad_output.slice(2, start_index, start_index + num_elements)
                            .reshape({B, Q, H, W});
        
        const int total_elements = B * Q * H * W;
        const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        // Temporary tensors for accumulation
        auto grad_queries_level = torch::zeros_like(queries);
        auto grad_rel_bias_level = torch::zeros_like(rel_bias);
        
        relative_attention_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            grad_level.data_ptr<float>(),
            queries.data_ptr<float>(),
            pos.data_ptr<float>(),
            rel_bias.data_ptr<float>(),
            grad_queries_level.data_ptr<float>(),
            grad_rel_bias_level.data_ptr<float>(),
            B, Q, C, H, W, H_rel, W_rel
        );

        cudaDeviceSynchronize();
        
        // Accumulate gradients
        grad_queries += grad_queries_level;
        cudaDeviceSynchronize();

        grad_rel_bias += grad_rel_bias_level;
        
        cudaDeviceSynchronize();
    }
    
    return {grad_queries, grad_keys, grad_rel_bias};
}

PYBIND11_MODULE(relative_grid_attn, m) {
    m.def("forward", &fused_attn_forward, "Fused Attention Forward (Multi-Resolution)");
    m.def("backward", &fused_attn_backward, "Fused Attention Backward (Multi-Resolution)");
}