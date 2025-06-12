#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int THREADS_PER_BLOCK = 512;

__global__ void fused_attn_forward_kernel(
    const float* __restrict__ queries,   // [B, Q, C]
    const float* __restrict__ keys,      // [B, H, W, C]
    const float* __restrict__ pos,       // [B, Q, 4]
    const float* __restrict__ rel_bias,  // [H_rel, W_rel, C]
    float* __restrict__ output,          // [B, Q, H, W]
    const int B,
    const int Q,
    const int C,
    const int H,
    const int W,
    const int H_rel,
    const int W_rel,
    const float scale
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
    const float* cur_key = &keys[b * H * W * C + h * W * C + w * C];
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute relative coordinates
    const float gx = ((float)w)/((float)(W-1));
    const float gy = ((float)h)/((float)(H-1));
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
    
    // Check boundaries once
    const bool in00 = (i0 >= 0) && (i0 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in01 = (i0 >= 0) && (i0 < H_rel) && (j1 >= 0) && (j1 < W_rel);
    const bool in10 = (i1 >= 0) && (i1 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in11 = (i1 >= 0) && (i1 < H_rel) && (j1 >= 0) && (j1 < W_rel);

    // Precompute rel_bias offsets
    const int offset00 = in00 ? (i0 * W_rel * C + j0 * C) : -1;
    const int offset01 = in01 ? (i0 * W_rel * C + j1 * C) : -1;
    const int offset10 = in10 ? (i1 * W_rel * C + j0 * C) : -1;
    const int offset11 = in11 ? (i1 * W_rel * C + j1 * C) : -1;

    // Fused loop for content and relative attention
    float content_val = 0.0f;
    float rel_val = 0.0f;
    for (int c = 0; c < C; c++) {
        // Load once per channel
        const float q_val = cur_query[c];
        const float k_val = cur_key[c];
        
        // Content attention
        content_val += q_val * k_val;
        
        // Relative attention
        float bias_val = 0.0f;
        if (in00) bias_val += w00 * rel_bias[offset00 + c];
        if (in01) bias_val += w01 * rel_bias[offset01 + c];
        if (in10) bias_val += w10 * rel_bias[offset10 + c];
        if (in11) bias_val += w11 * rel_bias[offset11 + c];
        rel_val += q_val * bias_val;
    }
    content_val *= scale;
    output[idx] = content_val + rel_val;
}

__global__ void fused_attn_backward_kernel(
    const float* __restrict__ grad_out,   // [B, Q, H, W]
    const float* __restrict__ queries,    // [B, Q, C]
    const float* __restrict__ keys,       // [B, H, W, C]
    const float* __restrict__ pos,        // [B, Q, 4]
    const float* __restrict__ rel_bias,   // [H_rel, W_rel, C]
    float* __restrict__ grad_queries,     // [B, Q, C]
    float* __restrict__ grad_keys,        // [B, H, W, C]
    float* __restrict__ grad_rel_bias,    // [H_rel, W_rel, C]
    const int B,
    const int Q,
    const int C,
    const int H,
    const int W,
    const int H_rel,
    const int W_rel,
    const float scale
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
    const int key_idx = b * H * W * C + h * W * C + w * C;
    
    // Get position data
    const float x_center = pos[b * Q * 4 + q * 4];
    const float y_center = pos[b * Q * 4 + q * 4 + 1];
    const float width = pos[b * Q * 4 + q * 4 + 2];
    const float height = pos[b * Q * 4 + q * 4 + 3];
    
    // Compute relative coordinates
    const float gx = ((float)w)/((float)(W-1));
    const float gy = ((float)h)/((float)(H-1));
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
    
    // Check boundaries once
    const bool in00 = (i0 >= 0) && (i0 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in01 = (i0 >= 0) && (i0 < H_rel) && (j1 >= 0) && (j1 < W_rel);
    const bool in10 = (i1 >= 0) && (i1 < H_rel) && (j0 >= 0) && (j0 < W_rel);
    const bool in11 = (i1 >= 0) && (i1 < H_rel) && (j1 >= 0) && (j1 < W_rel);

    // Precompute rel_bias offsets
    const int offset00 = in00 ? (i0 * W_rel * C + j0 * C) : -1;
    const int offset01 = in01 ? (i0 * W_rel * C + j1 * C) : -1;
    const int offset10 = in10 ? (i1 * W_rel * C + j0 * C) : -1;
    const int offset11 = in11 ? (i1 * W_rel * C + j1 * C) : -1;

    // Precompute interpolation weights for gradient updates
    const float dL_w00 = dL * w00;
    const float dL_w01 = dL * w01;
    const float dL_w10 = dL * w10;
    const float dL_w11 = dL * w11;

    // Fused loop for gradients
    for (int c = 0; c < C; c++) {
        const float q_val = queries[query_idx + c];
        const float k_val = keys[key_idx + c];
        const float grad_scale = dL * scale;

        // Content gradients
        atomicAdd(&grad_queries[query_idx + c], grad_scale * k_val);
        atomicAdd(&grad_keys[key_idx + c], grad_scale * q_val);

        // Relative gradients for queries and rel_bias
        float interp_val = 0.0f;
        if (in00) {
            const float rb_val = rel_bias[offset00 + c];
            interp_val += w00 * rb_val;
            atomicAdd(&grad_rel_bias[offset00 + c], dL_w00 * q_val);
        }
        if (in01) {
            const float rb_val = rel_bias[offset01 + c];
            interp_val += w01 * rb_val;
            atomicAdd(&grad_rel_bias[offset01 + c], dL_w01 * q_val);
        }
        if (in10) {
            const float rb_val = rel_bias[offset10 + c];
            interp_val += w10 * rb_val;
            atomicAdd(&grad_rel_bias[offset10 + c], dL_w10 * q_val);
        }
        if (in11) {
            const float rb_val = rel_bias[offset11 + c];
            interp_val += w11 * rb_val;
            atomicAdd(&grad_rel_bias[offset11 + c], dL_w11 * q_val);
        }
        atomicAdd(&grad_queries[query_idx + c], dL * interp_val);
    }
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
    
    auto options = torch::TensorOptions()
        .dtype(queries.dtype())
        .device(queries.device());
    auto output = torch::zeros({B, Q, H, W}, options);
    
    const int total_elements = B * Q * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const float scale = 1.0f / sqrtf(C);
    
    fused_attn_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        pos.data_ptr<float>(),
        rel_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, Q, C, H, W, H_rel, W_rel,
        scale
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
    
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);
    auto grad_rel_bias = torch::zeros_like(rel_bias);
    
    const int total_elements = B * Q * H * W;
    const int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const float scale = 1.0f / sqrtf(C);
    
    fused_attn_backward_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        grad_out.data_ptr<float>(),
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        pos.data_ptr<float>(),
        rel_bias.data_ptr<float>(),
        grad_queries.data_ptr<float>(),
        grad_keys.data_ptr<float>(),
        grad_rel_bias.data_ptr<float>(),
        B, Q, C, H, W, H_rel, W_rel,
        scale
    );
    
    return {grad_queries, grad_keys, grad_rel_bias};
}

PYBIND11_MODULE(relative_grid_attn, m) {
    m.def("forward", &fused_attn_forward, "Fused Attention Forward");
    m.def("backward", &fused_attn_backward, "Fused Attention Backward");
}