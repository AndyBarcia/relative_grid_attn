from torch.profiler import profile, record_function, ProfilerActivity
import torch
import time
import copy
import gc
import psutil
import os
from functions import RelativeGridAttnCUDAFunction, relative_grid_attn_python, RelativeGridAttention


def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
    else:
        gpu_memory = gpu_memory_cached = 0
    
    # CPU memory usage
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    
    return {
        'gpu_allocated': gpu_memory,
        'gpu_cached': gpu_memory_cached,
        'cpu_memory': cpu_memory
    }

def print_memory_usage(label, mem_info):
    """Print formatted memory usage information"""
    print(f"{label}:")
    print(f"  CPU Memory: {mem_info['cpu_memory']:.2f} MB")
    if torch.cuda.is_available():
        print(f"  GPU Allocated: {mem_info['gpu_allocated']:.2f} MB")
        print(f"  GPU Cached: {mem_info['gpu_cached']:.2f} MB")

def run_test():
    # Parameters
    B, Q, C = 4, 16, 64
    C_inner = 1
    H, W = 128, 128
    H_rel, W_rel = 32, 32
    dtype = torch.float32
    
    # Determine device: Use CUDA if extension is available AND torch.cuda is available
    device = torch.device("cuda")
    print(f"Testing with: B={B}, Q={Q}, C={C}, H={H}, W={W}, H_rel={H_rel}, W_rel={W_rel}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}, dtype: {dtype}\n")
    
    # Clear memory and get baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    baseline_memory = get_memory_info()
    print_memory_usage("Baseline Memory", baseline_memory)
    print()
    
    # Create random input tensors
    queries = torch.randn(B, Q, C, device=device, dtype=dtype)
    keys = torch.randn(B, H, W, C, device=device, dtype=dtype)
    pos_xy = torch.rand(B, Q, 2, device=device, dtype=dtype)
    pos_wh = torch.rand(B, Q, 2, device=device, dtype=dtype) * 0.5 + 0.1 # Ensure width/height > 0
    pos = torch.cat([pos_xy, pos_wh], dim=-1)

    # Create module to be tested.
    py_module = RelativeGridAttention(
        dim=C,
        res=(H_rel, W_rel),
        hidden_dim=C_inner,
        kernel_size=3,
        implementation="python"
    ).to(device)
    cuda_module = copy.deepcopy(py_module)
    cuda_module.implementation = "cuda"
    
    after_tensor_creation = get_memory_info()
    print_memory_usage("After Tensor Creation", after_tensor_creation)
    print()
    
    # Make copies for gradient checking
    queries_py = queries.clone().requires_grad_(True)
    keys_py = keys.clone().requires_grad_(True)
    pos_py = pos.clone().requires_grad_(False)
    
    # --- Forward and Backward for PyTorch implementation ---
    print("--- PyTorch Implementation ---")
        
    # PyTorch Profiler for detailed analysis
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("pytorch_forward"):
            output_py = py_module(queries_py, keys_py, pos_py)
            torch.cuda.synchronize()
    
    print("PYTORCH FORWARD PROFILER RESULTS:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    loss_py = output_py.sum()
    grad_output_val = torch.randn_like(output_py)
        
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof_bwd:
        with record_function("pytorch_backward"):
            loss_py.backward()
            torch.cuda.synchronize()
    
    print("PYTORCH BACKWARD PROFILER RESULTS:")
    print(prof_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # --- Forward and Backward for CUDA implementation (if available) ---
    print("--- CUDA Implementation ---")
    
    # Reset peak memory stats for CUDA implementation
    torch.cuda.reset_peak_memory_stats()
    
    queries_cuda = queries.clone().requires_grad_(True)
    keys_cuda = keys.clone().requires_grad_(True)
    pos_cuda = pos.clone().requires_grad_(False)
   
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof_cuda:
        with record_function("cuda_forward"):
            output_cuda = cuda_module(queries_cuda, keys_cuda, pos_cuda)
            torch.cuda.synchronize()
    
    print("CUDA FORWARD PROFILER RESULTS:")
    print(prof_cuda.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    loss_cuda = output_cuda.sum()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof_cuda_bwd:
        with record_function("cuda_backward"):
            loss_cuda.backward()
            torch.cuda.synchronize()
    
    print("CUDA BACKWARD PROFILER RESULTS:")
    print(prof_cuda_bwd.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n--- Numerical Accuracy Comparison ---")
    fwd_atol, fwd_rtol = 1e-5, 1e-4
    bwd_atol, bwd_rtol = 1e-4, 1e-3
    
    forward_pass_ok = torch.allclose(output_cuda, output_py.to(device), atol=fwd_atol, rtol=fwd_rtol)
    print(f"Forward outputs close: {forward_pass_ok}")
    if not forward_pass_ok:
        print("Max diff fwd (abs):", (output_cuda - output_py.to(device)).abs().max().item())
        print("Mean diff fwd (abs):", (output_cuda - output_py.to(device)).abs().mean().item())
        print("Max diff fwd (rel):", ((output_cuda - output_py.to(device)).abs() / (output_py.to(device).abs() + 1e-9)).max().item())
    
    grad_queries_ok = torch.allclose(queries_cuda.grad, queries_py.grad.to(device), atol=bwd_atol, rtol=bwd_rtol)
    print(f"Gradient for queries close: {grad_queries_ok}")
    if not grad_queries_ok:
        print("Max diff grad_queries (abs):", (queries_cuda.grad - queries_py.grad.to(device)).abs().max().item())
        print("Mean diff grad_queries (abs):", (queries_cuda.grad - queries_py.grad.to(device)).abs().mean().item())
    
    grad_keys_ok = torch.allclose(keys_cuda.grad, keys_py.grad.to(device), atol=bwd_atol, rtol=bwd_rtol)
    print(f"Gradient for keys close: {grad_keys_ok}")
    if not grad_keys_ok:
        print("Max diff grad_keys (abs):", (keys_cuda.grad - keys_py.grad.to(device)).abs().max().item())
        print("Mean diff grad_keys (abs):", (keys_cuda.grad - keys_py.grad.to(device)).abs().mean().item())
    
    grad_rel_bias_ok = torch.allclose(cuda_module.rel_bias.grad, py_module.rel_bias.grad.to(device), atol=bwd_atol, rtol=bwd_rtol)
    print(f"Gradient for rel_bias close: {grad_rel_bias_ok}")
    if not grad_rel_bias_ok:
        print("Max diff grad_rel_bias (abs):", (cuda_module.rel_bias.grad - py_module.rel_bias.grad.to(device)).abs().max().item())
        print("Mean diff grad_rel_bias (abs):", (cuda_module.rel_bias.grad - py_module.rel_bias.grad.to(device)).abs().mean().item())
    
    if forward_pass_ok and grad_queries_ok and grad_keys_ok and grad_rel_bias_ok:
        print("\nAll CUDA vs PyTorch checks passed!")
    else:
        print("\nSome CUDA vs PyTorch checks FAILED.")
        
    # Optionally save profiler traces
    print("\n--- Profiler Traces (saved to files) ---")
    prof.export_chrome_trace("pytorch_profile.json")
    prof_cuda.export_chrome_trace("cuda_profile.json")
    print("PyTorch profiler trace saved to: pytorch_profile.json")
    print("CUDA profiler trace saved to: cuda_profile.json")
    print("View these files in chrome://tracing/ for detailed analysis")


if __name__ == "__main__":
    # Ensure setup.py has been run to compile fused_attn_ext
    # e.g., python setup.py build_ext --inplace
    run_test()