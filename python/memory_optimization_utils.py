#!/usr/bin/env python3
"""
Memory optimization utilities for large-scale data processing
"""

import pandas as pd
import numpy as np
import psutil
import gc
from typing import Generator, Optional, Dict, Any
import warnings

class MemoryOptimizer:
    """Memory optimization utilities for data processing"""
    
    def __init__(self, max_memory_usage: float = 0.8):
        """
        Initialize memory optimizer
        
        Args:
            max_memory_usage: Maximum memory usage as fraction of available memory
        """
        self.max_memory_usage = max_memory_usage
        self.available_memory = psutil.virtual_memory().available
        self.max_memory_bytes = int(self.available_memory * max_memory_usage)
    
    def get_optimal_chunk_size(self, df_sample: pd.DataFrame, 
                              target_memory_mb: Optional[int] = None) -> int:
        """
        Calculate optimal chunk size based on memory usage
        
        Args:
            df_sample: Sample DataFrame to estimate memory usage
            target_memory_mb: Target memory usage in MB (optional)
        
        Returns:
            Optimal chunk size
        """
        if target_memory_mb is None:
            target_memory_bytes = self.max_memory_bytes
        else:
            target_memory_bytes = target_memory_mb * 1024 * 1024
        
        # Estimate memory usage per row
        memory_per_row = df_sample.memory_usage(deep=True).sum() / len(df_sample)
        
        # Calculate chunk size
        chunk_size = int(target_memory_bytes / memory_per_row)
        
        # Ensure reasonable bounds
        chunk_size = max(1000, min(chunk_size, 1000000))
        
        return chunk_size
    
    def optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with optimized dtypes
        """
        df_optimized = df.copy()
        
        # Optimize integer columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            if df_optimized[col].min() >= 0:
                if df_optimized[col].max() < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif df_optimized[col].max() < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif df_optimized[col].max() < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if df_optimized[col].min() > -128 and df_optimized[col].max() < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif df_optimized[col].min() > -32768 and df_optimized[col].max() < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif df_optimized[col].min() > -2147483648 and df_optimized[col].max() < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Optimize object columns (convert to category if beneficial)
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage
        
        Returns:
            Dictionary with memory usage statistics
        """
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent
        }
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
    
    def process_in_memory_chunks(self, df: pd.DataFrame, 
                                chunk_size: int,
                                process_func,
                                *args, **kwargs) -> Generator[pd.DataFrame, None, None]:
        """
        Process DataFrame in memory-efficient chunks
        
        Args:
            df: Input DataFrame
            chunk_size: Size of each chunk
            process_func: Function to process each chunk
            *args, **kwargs: Arguments to pass to process_func
        
        Yields:
            Processed DataFrame chunks
        """
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            # Process chunk
            processed_chunk = process_func(chunk, *args, **kwargs)
            
            # Yield processed chunk
            yield processed_chunk
            
            # Force garbage collection
            del chunk
            self.force_garbage_collection()
            
            # Check memory usage
            memory_stats = self.monitor_memory_usage()
            if memory_stats['usage_percent'] > self.max_memory_usage * 100:
                warnings.warn(f"High memory usage: {memory_stats['usage_percent']:.1f}%")

def create_memory_efficient_reader(file_path: str, 
                                  chunk_size: int = 10000,
                                  **kwargs) -> Generator[pd.DataFrame, None, None]:
    """
    Create memory-efficient file reader
    
    Args:
        file_path: Path to file
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for pandas reader
    
    Yields:
        DataFrame chunks
    """
    if file_path.endswith('.csv'):
        reader = pd.read_csv(file_path, chunksize=chunk_size, **kwargs)
    elif file_path.endswith(('.xlsx', '.xls')):
        reader = pd.read_excel(file_path, chunksize=chunk_size, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    for chunk in reader:
        yield chunk

def optimize_sql_query_memory(query: str, 
                             conn,
                             chunk_size: int = 50000,
                             **kwargs) -> Generator[pd.DataFrame, None, None]:
    """
    Optimize SQL query for memory usage
    
    Args:
        query: SQL query string
        conn: Database connection
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for pandas.read_sql_query
    
    Yields:
        DataFrame chunks
    """
    return pd.read_sql_query(query, conn, chunksize=chunk_size, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test memory optimizer
    optimizer = MemoryOptimizer()
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'id': range(1000),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Get optimal chunk size
    chunk_size = optimizer.get_optimal_chunk_size(sample_df)
    print(f"Optimal chunk size: {chunk_size}")
    
    # Optimize dtypes
    optimized_df = optimizer.optimize_dataframe_dtypes(sample_df)
    print(f"Memory reduction: {sample_df.memory_usage().sum() / optimized_df.memory_usage().sum():.2f}x")
    
    # Monitor memory
    memory_stats = optimizer.monitor_memory_usage()
    print(f"Memory usage: {memory_stats['usage_percent']:.1f}%")

