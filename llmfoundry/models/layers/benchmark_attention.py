#!/usr/bin/env python3
"""
Benchmark script for GroupedQueryAttention with different attention implementations.

This script benchmarks the forward and backward pass performance of GroupedQueryAttention
using different attention implementations (torch, flash, native_sparse_attention).
"""

import argparse
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
import statistics

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install with: pip install torch")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Please install with: pip install numpy")



# Import the attention classes
import sys
import os

# Add the llm-foundry root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
llm_foundry_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if llm_foundry_root not in sys.path:
    sys.path.insert(0, llm_foundry_root)

try:
    # Try importing attention classes from llmfoundry package
    from llmfoundry.models.layers.attention import (
        GroupedQueryAttention,
        MultiheadAttention,
        MultiQueryAttention,
        NativeSparseAttention,
        NativeSparseAttention2,
        is_flash_v2_installed,
        is_flash_v1_installed,
    )
    ATTENTION_AVAILABLE = True
    print("✓ Attention classes imported successfully from llmfoundry package")
except ImportError as e:
    print(f"✗ Could not import from llmfoundry package: {e}")
    
    # Fallback: try direct import (this will likely fail due to missing dependencies)
    try:
        # Add current directory to path for direct imports
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from attention import (
            GroupedQueryAttention,
            MultiheadAttention,
            MultiQueryAttention,
            NativeSparseAttention2,
            is_flash_v2_installed,
            is_flash_v1_installed,
        )
        ATTENTION_AVAILABLE = True
        print("✓ Attention classes imported successfully (direct import)")
    except ImportError as e2:
        ATTENTION_AVAILABLE = False
        print(f"✗ Could not import attention classes: {e2}")
        
        # Check if attention.py exists
        attention_file = os.path.join(current_dir, "attention.py")
        if os.path.exists(attention_file):
            print(f"✓ attention.py found at: {attention_file}")
            print("✗ Import failed - this is likely due to missing llmfoundry dependencies")
            print("  Solution: Run from the llm-foundry root directory, or install missing dependencies")
        else:
            print(f"✗ attention.py not found at: {attention_file}")
            print("Make sure you're running from the correct directory.")


class BenchmarkConfig:
    """Configuration for attention benchmarking."""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        kv_n_heads: int = 2,
        batch_size: int = 4,
        seq_len: int = 512,
        head_dim: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_warmup: int = 10,
        num_iterations: int = 50,
        use_kv_cache: bool = False,
        causal: bool = True,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_n_heads = kv_n_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.head_dim = head_dim or d_model // n_heads
        self.device = device
        self.dtype = dtype
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.use_kv_cache = use_kv_cache
        self.causal = causal


class AttentionBenchmark:
    """Benchmark suite for attention implementations."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Check available implementations
        self.available_impls = self._check_available_implementations()
        print(f"Available attention implementations: {self.available_impls}")
    
    def _check_available_implementations(self) -> List[str]:
        """Check which attention implementations are available."""
        available = []  # torch is always available
        
        if is_flash_v1_installed() or is_flash_v2_installed():
            available.append('flash')
        
        return available
    
    def create_attention_layer(self, attn_impl: str, attention_class: str = "grouped_query_attention") -> nn.Module:
        """Create an attention layer with the specified implementation."""
        common_kwargs = {
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'head_dim': self.config.head_dim,
            'attn_impl': attn_impl,
            'device': self.config.device,
            'bias': True,
            'attention_bias': True,
            'attn_pdrop': 0.0,
        }
        
        if attention_class == "grouped_query_attention":
            return GroupedQueryAttention(
                kv_n_heads=self.config.kv_n_heads,
                **common_kwargs
            )
        elif attention_class == "multihead_attention":
            return MultiheadAttention(**common_kwargs)
        elif attention_class == "multiquery_attention":
            return MultiQueryAttention(**common_kwargs)
        elif attention_class == "native_sparse_attention":
            return NativeSparseAttention(
                kv_n_heads=self.config.kv_n_heads,
                **common_kwargs
            )
        elif attention_class == "native_sparse_attention2":
            return NativeSparseAttention2(
                kv_n_heads=self.config.kv_n_heads,
                **common_kwargs
            )
        else:
            raise ValueError(f"Unknown attention class: {attention_class}")
    
    def create_test_inputs(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Create test inputs for the attention layer."""
        x = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.d_model,
            device=self.device,
            dtype=self.config.dtype,
            requires_grad=True
        )
        
        kwargs = {
            'is_causal': self.config.causal,
            'needs_weights': False,
        }
        
        # Create flash attention padding info (required for flash attention)
        batch_size, seq_len = self.config.batch_size, self.config.seq_len
        
        # For simplicity, assume no padding (all sequences are full length)
        # In real usage, this would be computed based on actual sequence lengths
        indices = torch.arange(batch_size * seq_len, device=self.device)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, device=self.device, dtype=torch.int32)
        
        flash_attn_padding_info = {
            'indices_q': indices,
            'indices_k': indices, 
            'indices_v': indices,
            'cu_seqlens_q': cu_seqlens,
            'cu_seqlens_k': cu_seqlens,
            'max_seqlen_q': seq_len,
            'max_seqlen_k': seq_len,
        }
        
        kwargs['flash_attn_padding_info'] = flash_attn_padding_info
        
        if self.config.use_kv_cache:
            # Simulate KV cache
            past_key_value = (
                torch.randn(
                    self.config.batch_size,
                    self.config.kv_n_heads,
                    64,  # past sequence length
                    self.config.head_dim,
                    device=self.device,
                    dtype=self.config.dtype
                ),
                torch.randn(
                    self.config.batch_size,
                    self.config.kv_n_heads,
                    64,  # past sequence length
                    self.config.head_dim,
                    device=self.device,
                    dtype=self.config.dtype
                )
            )
            kwargs['past_key_value'] = past_key_value
        
        return x, kwargs
    
    def warmup_gpu(self):
        """Warmup the GPU to get stable timings."""
        warmup_tensor = torch.randn(1000, 1000, device=self.device)
        for _ in range(10):
            torch.mm(warmup_tensor, warmup_tensor)
        torch.cuda.synchronize()
    
    def benchmark_forward_pass(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        kwargs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Benchmark the forward pass of an attention layer."""
        model.eval()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            with torch.no_grad():
                _ = model(x, **kwargs)
        torch.cuda.synchronize()
        
        # Timing
        times = []
        for _ in range(self.config.num_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(x, **kwargs)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_ms': statistics.mean(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_ms': min(times),
            'max_ms': max(times),
            'median_ms': statistics.median(times),
        }
    
    def benchmark_backward_pass(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        kwargs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Benchmark the backward pass of an attention layer."""
        model.train()
        
        # Warmup
        for _ in range(self.config.num_warmup):
            x_copy = x.clone().detach().requires_grad_(True)
            output, _, _ = model(x_copy, **kwargs)
            loss = output.sum()
            loss.backward()
        torch.cuda.synchronize()
        
        # Timing
        times = []
        for _ in range(self.config.num_iterations):
            x_copy = x.clone().detach().requires_grad_(True)
            
            # Forward pass
            output, _, _ = model(x_copy, **kwargs)
            loss = output.sum()
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Backward pass
            loss.backward()
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_ms': statistics.mean(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'min_ms': min(times),
            'max_ms': max(times),
            'median_ms': statistics.median(times),
        }
    
    def benchmark_memory_usage(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        kwargs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Benchmark memory usage of an attention layer."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.train()
        x_copy = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        output, _, _ = model(x_copy, **kwargs)
        forward_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        total_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'forward_memory_mb': forward_memory,
            'total_memory_mb': total_memory,
            'backward_memory_mb': total_memory - forward_memory,
        }
    
    def run_comprehensive_benchmark(
        self, 
        attention_classes: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive benchmarks for all available implementations."""
        if attention_classes is None:
            attention_classes = ["native_sparse_attention", "native_sparse_attention2", "grouped_query_attention"]
        
        results = {}
        
        print(f"\nStarting benchmark with configuration:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Sequence length: {self.config.seq_len}")
        print(f"  Model dimension: {self.config.d_model}")
        print(f"  Number of heads: {self.config.n_heads}")
        print(f"  KV heads: {self.config.kv_n_heads}")
        print(f"  Head dimension: {self.config.head_dim}")
        print(f"  Device: {self.config.device}")
        print(f"  Data type: {self.config.dtype}")
        print(f"  Iterations: {self.config.num_iterations}")
        print("-" * 80)
        
        self.warmup_gpu()
        
        for attention_class in attention_classes:
            print(f"\nBenchmarking {attention_class}...")
            results[attention_class] = {}
            
            for attn_impl in self.available_impls:
                print(f"  Testing {attn_impl} implementation...")
                
                try:
                    # Create model and inputs
                    model = self.create_attention_layer(attn_impl, attention_class)
                    model = model.to(self.device).to(self.config.dtype)
                    x, kwargs = self.create_test_inputs()
                    
                    # For torch implementation, remove flash-specific parameters
                    if attn_impl == 'torch':
                        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'flash_attn_padding_info'}
                    else:
                        kwargs_filtered = kwargs
                    
                    # Test a single forward pass first to catch issues early
                    model.eval()
                    with torch.no_grad():
                        test_output = model(x, **kwargs_filtered)
                    
                    # Run benchmarks
                    forward_stats = self.benchmark_forward_pass(model, x, kwargs_filtered)
                    backward_stats = self.benchmark_backward_pass(model, x, kwargs_filtered)
                    memory_stats = self.benchmark_memory_usage(model, x, kwargs_filtered)
                    
                    results[attention_class][attn_impl] = {
                        'forward_pass': forward_stats,
                        'backward_pass': backward_stats,
                        'memory_usage': memory_stats,
                    }
                    
                    print(f"    Forward:  {forward_stats['mean_ms']:.3f} ± {forward_stats['std_ms']:.3f} ms")
                    print(f"    Backward: {backward_stats['mean_ms']:.3f} ± {backward_stats['std_ms']:.3f} ms")
                    print(f"    Memory:   {memory_stats['total_memory_mb']:.1f} MB")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    results[attention_class][attn_impl] = {'error': str(e)}
        
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save benchmark results to a file."""
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attention_benchmark_{timestamp}.json"
        
        # Add configuration to results
        results_with_config = {
            'config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'kv_n_heads': self.config.kv_n_heads,
                'batch_size': self.config.batch_size,
                'seq_len': self.config.seq_len,
                'head_dim': self.config.head_dim,
                'device': self.config.device,
                'dtype': str(self.config.dtype),
                'num_iterations': self.config.num_iterations,
                'use_kv_cache': self.config.use_kv_cache,
                'causal': self.config.causal,
            },
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(filename, 'w') as f:
            json.dump(results_with_config, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_comparison_table(self, results: Dict[str, Dict[str, Any]]):
        """Print a comparison table of results."""
        print("\n" + "="*120)
        print("PERFORMANCE COMPARISON TABLE")
        print("="*120)
        
        # Header
        print(f"{'Attention Class':<25} {'Implementation':<15} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'Memory (MB)':<15}")
        print("-" * 120)
        
        for attention_class, impls in results.items():
            for impl, stats in impls.items():
                if 'error' in stats:
                    print(f"{attention_class:<25} {impl:<15} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
                else:
                    forward_time = stats['forward_pass']['mean_ms']
                    backward_time = stats['backward_pass']['mean_ms']
                    total_time = forward_time + backward_time
                    memory = stats['memory_usage']['total_memory_mb']
                    print(f"{attention_class:<25} {impl:<15} {forward_time:<15.3f} {backward_time:<15.3f} {total_time:<15.3f} {memory:<15.1f}")
        
        print("="*120)


def main():
    """Main function to run the benchmark."""
    # Check dependencies
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required. Please install with: pip install torch")
        return 1
    
    if not ATTENTION_AVAILABLE:
        print("Error: Could not import attention classes. Please ensure you're in the correct directory.")
        return 1
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Benchmarks will not be meaningful on CPU.")
    
    parser = argparse.ArgumentParser(description="Benchmark GroupedQueryAttention implementations")
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=64, help="Number of query heads")
    parser.add_argument("--kv_n_heads", type=int, default=4, help="Number of key-value heads")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=32768, help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations for timing")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--use_kv_cache", action="store_true", help="Test with KV cache")
    parser.add_argument("--attention_classes", nargs="+", default=None, help="Attention classes to test")
    # first one is b, 
    # second is seq_len (max length of each sequence, not total)
    # 
    # third one is d_model for x and q 
    # head_dim * num_kv_heads for k and v 
    ##
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Create configuration
    config = BenchmarkConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        kv_n_heads=args.kv_n_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        dtype=dtype,
        num_iterations=args.iterations,
        num_warmup=args.warmup,
        use_kv_cache=args.use_kv_cache,
    )
    
    # Run benchmark
    benchmark = AttentionBenchmark(config)
    results = benchmark.run_comprehensive_benchmark(attention_classes=args.attention_classes)
    benchmark.print_comparison_table(results)


if __name__ == "__main__":
    main()
