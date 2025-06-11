from torch.profiler import profile, record_function, ProfilerActivity
import torch
import time
import gc
import psutil
import os
from functions import RelativeGridAttnCUDAFunction, relative_grid_attn_python

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
    B, Q, C = 32, 300, 256
    H, W = 16, 16
    H_rel, W_rel = 32, 32
    dtype = torch.float32
    
    # Determine device: Use CUDA if extension is available AND torch.cuda is available
    use_cuda_device = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda_device else "cpu")
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
    rel_bias = torch.randn(H_rel, W_rel, C, device=device, dtype=dtype)
    grid_y_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
    grid_x_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(grid_x_coords, grid_y_coords, indexing='xy')
    grid_x = grid_x.contiguous()
    grid_y = grid_y.contiguous()
    
    after_tensor_creation = get_memory_info()
    print_memory_usage("After Tensor Creation", after_tensor_creation)
    print()
    
    # Make copies for gradient checking
    queries_py = queries.clone().requires_grad_(True)
    keys_py = keys.clone().requires_grad_(True)
    pos_py = pos.clone().requires_grad_(False)
    rel_bias_py = rel_bias.clone().requires_grad_(True)
    grid_x_py = grid_x.clone().requires_grad_(False)
    grid_y_py = grid_y.clone().requires_grad_(False)
    
    # --- Forward and Backward for PyTorch implementation ---
    print("--- PyTorch Implementation ---")
    
    # Memory before PyTorch forward
    before_py_fwd = get_memory_info()
    
    # PyTorch Profiler for detailed analysis
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if use_cuda_device else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("pytorch_forward"):
            start_time = time.time()
            output_py = relative_grid_attn_python(
                queries_py, keys_py, pos_py, rel_bias_py, grid_x_py, grid_y_py,
            )
            if device.type == 'cuda': 
                torch.cuda.synchronize()
            py_fwd_time = time.time() - start_time
    
    after_py_fwd = get_memory_info()
    print(f"PyTorch forward time: {py_fwd_time:.6f} s")
    print_memory_usage("After PyTorch Forward", after_py_fwd)
    
    # Memory difference for forward pass
    py_fwd_memory_diff = {
        'gpu_allocated': after_py_fwd['gpu_allocated'] - before_py_fwd['gpu_allocated'],
        'gpu_cached': after_py_fwd['gpu_cached'] - before_py_fwd['gpu_cached'],
        'cpu_memory': after_py_fwd['cpu_memory'] - before_py_fwd['cpu_memory']
    }
    print("PyTorch Forward Memory Usage:")
    print(f"  CPU Memory Delta: {py_fwd_memory_diff['cpu_memory']:.2f} MB")
    if torch.cuda.is_available():
        print(f"  GPU Allocated Delta: {py_fwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  GPU Cached Delta: {py_fwd_memory_diff['gpu_cached']:.2f} MB")
    
    grad_output_val = torch.randn_like(output_py)
    
    # Memory before PyTorch backward
    before_py_bwd = get_memory_info()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if use_cuda_device else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof_bwd:
        with record_function("pytorch_backward"):
            start_time = time.time()
            output_py.backward(grad_output_val.clone())
            if device.type == 'cuda': 
                torch.cuda.synchronize()
            py_bwd_time = time.time() - start_time
    
    after_py_bwd = get_memory_info()
    print(f"PyTorch backward time: {py_bwd_time:.6f} s")
    print_memory_usage("After PyTorch Backward", after_py_bwd)
    
    # Memory difference for backward pass
    py_bwd_memory_diff = {
        'gpu_allocated': after_py_bwd['gpu_allocated'] - before_py_bwd['gpu_allocated'],
        'gpu_cached': after_py_bwd['gpu_cached'] - before_py_bwd['gpu_cached'],
        'cpu_memory': after_py_bwd['cpu_memory'] - before_py_bwd['cpu_memory']
    }
    print("PyTorch Backward Memory Usage:")
    print(f"  CPU Memory Delta: {py_bwd_memory_diff['cpu_memory']:.2f} MB")
    if torch.cuda.is_available():
        print(f"  GPU Allocated Delta: {py_bwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  GPU Cached Delta: {py_bwd_memory_diff['gpu_cached']:.2f} MB")
        print(f"  GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print()
    
    # --- Forward and Backward for CUDA implementation (if available) ---
    if use_cuda_device:
        print("--- CUDA Implementation ---")
        
        # Reset peak memory stats for CUDA implementation
        torch.cuda.reset_peak_memory_stats()
        
        queries_cuda = queries.clone().requires_grad_(True)
        keys_cuda = keys.clone().requires_grad_(True)
        pos_cuda = pos.clone().requires_grad_(False)
        rel_bias_cuda = rel_bias.clone().requires_grad_(True)
        grid_x_cuda = grid_x.clone().requires_grad_(False)
        grid_y_cuda = grid_y.clone().requires_grad_(False)
        
        # Memory before CUDA forward
        before_cuda_fwd = get_memory_info()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof_cuda:
            with record_function("cuda_forward"):
                start_time = time.time()
                output_cuda = RelativeGridAttnCUDAFunction.apply(
                    queries_cuda, keys_cuda, pos_cuda, rel_bias_cuda, grid_x_cuda, grid_y_cuda,
                )
                torch.cuda.synchronize()
                cuda_fwd_time = time.time() - start_time
        
        after_cuda_fwd = get_memory_info()
        print(f"CUDA forward time: {cuda_fwd_time:.6f} s")
        print_memory_usage("After CUDA Forward", after_cuda_fwd)
        
        # Memory difference for CUDA forward pass
        cuda_fwd_memory_diff = {
            'gpu_allocated': after_cuda_fwd['gpu_allocated'] - before_cuda_fwd['gpu_allocated'],
            'gpu_cached': after_cuda_fwd['gpu_cached'] - before_cuda_fwd['gpu_cached'],
            'cpu_memory': after_cuda_fwd['cpu_memory'] - before_cuda_fwd['cpu_memory']
        }
        print("CUDA Forward Memory Usage:")
        print(f"  CPU Memory Delta: {cuda_fwd_memory_diff['cpu_memory']:.2f} MB")
        print(f"  GPU Allocated Delta: {cuda_fwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  GPU Cached Delta: {cuda_fwd_memory_diff['gpu_cached']:.2f} MB")
        
        # Memory before CUDA backward
        before_cuda_bwd = get_memory_info()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof_cuda_bwd:
            with record_function("cuda_backward"):
                start_time = time.time()
                output_cuda.backward(grad_output_val.clone())
                torch.cuda.synchronize()
                cuda_bwd_time = time.time() - start_time
        
        after_cuda_bwd = get_memory_info()
        print(f"CUDA backward time: {cuda_bwd_time:.6f} s")
        print_memory_usage("After CUDA Backward", after_cuda_bwd)
        
        # Memory difference for CUDA backward pass
        cuda_bwd_memory_diff = {
            'gpu_allocated': after_cuda_bwd['gpu_allocated'] - before_cuda_bwd['gpu_allocated'],
            'gpu_cached': after_cuda_bwd['gpu_cached'] - before_cuda_bwd['gpu_cached'],
            'cpu_memory': after_cuda_bwd['cpu_memory'] - before_cuda_bwd['cpu_memory']
        }
        print("CUDA Backward Memory Usage:")
        print(f"  CPU Memory Delta: {cuda_bwd_memory_diff['cpu_memory']:.2f} MB")
        print(f"  GPU Allocated Delta: {cuda_bwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  GPU Cached Delta: {cuda_bwd_memory_diff['gpu_cached']:.2f} MB")
        print(f"  GPU Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        # --- Comparisons ---
        print("\n--- Performance Comparison (CUDA vs PyTorch) ---")
        print(f"Forward speedup: {py_fwd_time / cuda_fwd_time:.2f}x")
        print(f"Backward speedup: {py_bwd_time / cuda_bwd_time:.2f}x")
        print(f"Total speedup: {(py_fwd_time + py_bwd_time) / (cuda_fwd_time + cuda_bwd_time):.2f}x")
        
        print("\n--- Memory Comparison (CUDA vs PyTorch) ---")
        print("Forward Pass Memory Usage:")
        print(f"  PyTorch GPU Delta: {py_fwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  CUDA GPU Delta: {cuda_fwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  Memory Efficiency (CUDA vs PyTorch): {py_fwd_memory_diff['gpu_allocated'] / cuda_fwd_memory_diff['gpu_allocated']:.2f}x" if cuda_fwd_memory_diff['gpu_allocated'] > 0 else "N/A")
        
        print("Backward Pass Memory Usage:")
        print(f"  PyTorch GPU Delta: {py_bwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  CUDA GPU Delta: {cuda_bwd_memory_diff['gpu_allocated']:.2f} MB")
        print(f"  Memory Efficiency (CUDA vs PyTorch): {py_bwd_memory_diff['gpu_allocated'] / cuda_bwd_memory_diff['gpu_allocated']:.2f}x" if cuda_bwd_memory_diff['gpu_allocated'] > 0 else "N/A")
        
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
        
        grad_rel_bias_ok = torch.allclose(rel_bias_cuda.grad, rel_bias_py.grad.to(device), atol=bwd_atol, rtol=bwd_rtol)
        print(f"Gradient for rel_bias close: {grad_rel_bias_ok}")
        if not grad_rel_bias_ok:
            print("Max diff grad_rel_bias (abs):", (rel_bias_cuda.grad - rel_bias_py.grad.to(device)).abs().max().item())
            print("Mean diff grad_rel_bias (abs):", (rel_bias_cuda.grad - rel_bias_py.grad.to(device)).abs().mean().item())
        
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
        
    else:
        print("\nCUDA implementation tests skipped (running on CPU, extension requires CUDA tensors).")

if __name__ == "__main__":
    # Ensure setup.py has been run to compile fused_attn_ext
    # e.g., python setup.py build_ext --inplace
    run_test()