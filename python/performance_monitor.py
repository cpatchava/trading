#!/usr/bin/env python3
"""
Performance monitoring and benchmarking utilities
"""

import time
import psutil
import pandas as pd
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import functools
import tracemalloc
# import line_profiler  # Optional dependency
# import memory_profiler  # Optional dependency

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
    
    @contextmanager
    def monitor(self, operation_name: str = "Operation"):
        """
        Context manager for monitoring performance
        
        Args:
            operation_name: Name of the operation being monitored
        """
        print(f"🚀 Starting {operation_name}...")
        
        # Start monitoring
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tracemalloc.start()
        
        try:
            yield self
        finally:
            # Stop monitoring
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            duration = end_time - self.start_time
            memory_delta = end_memory - self.start_memory
            peak_memory_mb = peak / 1024 / 1024
            
            # Print results
            print(f"✅ {operation_name} completed:")
            print(f"   ⏱️  Duration: {duration:.2f} seconds")
            print(f"   💾 Memory delta: {memory_delta:+.1f} MB")
            print(f"   📈 Peak memory: {peak_memory_mb:.1f} MB")
            print(f"   🔄 Memory efficiency: {memory_delta/duration:.1f} MB/s")
    
    def benchmark_function(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function's performance
        
        Args:
            func: Function to benchmark
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Dictionary with performance metrics
        """
        with self.monitor(f"Benchmarking {func.__name__}"):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                'function': func.__name__,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'result': result
            }

def profile_function(func):
    """
    Decorator to profile function performance
    
    Args:
        func: Function to profile
    
    Returns:
        Decorated function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        with monitor.monitor(f"Profiling {func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

def benchmark_dataframe_operations(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Benchmark different DataFrame operations
    
    Args:
        df: Input DataFrame
        operations: List of operations to benchmark
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    monitor = PerformanceMonitor()
    
    for op in operations:
        op_name = op['name']
        op_func = op['function']
        op_args = op.get('args', [])
        op_kwargs = op.get('kwargs', {})
        
        with monitor.monitor(op_name):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            result = op_func(df, *op_args, **op_kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results.append({
                'operation': op_name,
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'result_shape': getattr(result, 'shape', 'N/A'),
                'result_type': type(result).__name__
            })
    
    return pd.DataFrame(results)

def compare_implementations(implementations: List[Dict[str, Any]], 
                          test_data: Any,
                          iterations: int = 1) -> pd.DataFrame:
    """
    Compare different implementations of the same operation
    
    Args:
        implementations: List of implementation dictionaries
        test_data: Data to test with
        iterations: Number of iterations to run
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    monitor = PerformanceMonitor()
    
    for impl in implementations:
        impl_name = impl['name']
        impl_func = impl['function']
        impl_args = impl.get('args', [])
        impl_kwargs = impl.get('kwargs', {})
        
        durations = []
        memory_deltas = []
        
        for i in range(iterations):
            with monitor.monitor(f"{impl_name} (iteration {i+1})"):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                result = impl_func(test_data, *impl_args, **impl_kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                durations.append(end_time - start_time)
                memory_deltas.append(end_memory - start_memory)
        
        results.append({
            'implementation': impl_name,
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'avg_memory_delta': np.mean(memory_deltas),
            'std_memory_delta': np.std(memory_deltas),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations)
        })
    
    return pd.DataFrame(results)

def analyze_memory_usage(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Analyze memory usage of a function
    
    Args:
        func: Function to analyze
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Dictionary with memory analysis
    """
    # Start memory profiling
    tracemalloc.start()
    
    # Run function
    result = func(*args, **kwargs)
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'current_memory_mb': current / 1024 / 1024,
        'peak_memory_mb': peak / 1024 / 1024,
        'result': result
    }

# Example usage and testing
if __name__ == "__main__":
    # Test performance monitor
    monitor = PerformanceMonitor()
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C'], 100000),
        'value': np.random.randn(100000),
        'date': pd.date_range('2020-01-01', periods=100000, freq='D')
    })
    
    # Test different operations
    operations = [
        {
            'name': 'Groupby Sum',
            'function': lambda df: df.groupby('group')['value'].sum()
        },
        {
            'name': 'Rolling Mean',
            'function': lambda df: df['value'].rolling(window=100).mean()
        },
        {
            'name': 'Sort Values',
            'function': lambda df: df.sort_values('value')
        }
    ]
    
    # Benchmark operations
    benchmark_results = benchmark_dataframe_operations(sample_df, operations)
    print("Benchmark Results:")
    print(benchmark_results)
    
    # Test memory analysis
    def test_function(df):
        return df.groupby('group')['value'].sum()
    
    memory_analysis = analyze_memory_usage(test_function, sample_df)
    print(f"\nMemory Analysis:")
    print(f"Peak memory: {memory_analysis['peak_memory_mb']:.1f} MB")
