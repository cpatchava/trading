#!/usr/bin/env python3
"""
Advanced parallel processing utilities for data processing optimization
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import time
from typing import Callable, List, Any, Generator, Tuple
import sqlite3
from database_optimizations import OptimizedDBManager

class ParallelProcessor:
    """Advanced parallel processing utilities"""
    
    def __init__(self, max_workers: int = None, use_threads: bool = False):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            use_threads: Use threads instead of processes (for I/O bound tasks)
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    def process_dataframe_parallel(self, 
                                  df: pd.DataFrame,
                                  process_func: Callable,
                                  chunk_size: int = None,
                                  *args, **kwargs) -> pd.DataFrame:
        """
        Process DataFrame in parallel chunks
        
        Args:
            df: Input DataFrame
            process_func: Function to process each chunk
            chunk_size: Size of each chunk (auto-calculated if None)
            *args, **kwargs: Arguments to pass to process_func
        
        Returns:
            Processed DataFrame
        """
        if chunk_size is None:
            chunk_size = max(1000, len(df) // (self.max_workers * 4))
        
        # Split DataFrame into chunks
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_func, chunk, *args, **kwargs): chunk 
                for chunk in chunks
            }
            
            # Collect results
            results = []
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    # Add original chunk if processing failed
                    results.append(future_to_chunk[future])
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def process_database_parallel(self,
                                 db_path: str,
                                 query: str,
                                 process_func: Callable,
                                 chunk_size: int = 50000,
                                 *args, **kwargs) -> List[Any]:
        """
        Process database queries in parallel
        
        Args:
            db_path: Path to database
            query: SQL query to execute
            process_func: Function to process each chunk
            chunk_size: Size of each chunk
            *args, **kwargs: Arguments to pass to process_func
        
        Returns:
            List of processed results
        """
        # Get total count for progress tracking
        with OptimizedDBManager(db_path).get_connection() as conn:
            count_query = f"SELECT COUNT(*) FROM ({query})"
            total_count = conn.execute(count_query).fetchone()[0]
        
        # Create chunk queries
        chunk_queries = []
        for offset in range(0, total_count, chunk_size):
            chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
            chunk_queries.append(chunk_query)
        
        # Process chunks in parallel
        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_query = {
                executor.submit(self._process_chunk, db_path, query, process_func, *args, **kwargs): query
                for query in chunk_queries
            }
            
            # Collect results
            results = []
            for future in as_completed(future_to_query):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        
        return results
    
    def _process_chunk(self, db_path: str, query: str, process_func: Callable, *args, **kwargs):
        """Process a single database chunk"""
        with OptimizedDBManager(db_path).get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            return process_func(df, *args, **kwargs)
    
    def parallel_apply(self, 
                      df: pd.DataFrame,
                      func: Callable,
                      axis: int = 0,
                      *args, **kwargs) -> pd.DataFrame:
        """
        Parallel version of pandas apply
        
        Args:
            df: Input DataFrame
            func: Function to apply
            axis: Axis to apply function along
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            DataFrame with applied function
        """
        if axis == 0:  # Apply to columns
            columns = df.columns
            with self.executor_class(max_workers=self.max_workers) as executor:
                results = list(executor.map(
                    lambda col: func(df[col], *args, **kwargs), 
                    columns
                ))
            return pd.DataFrame(dict(zip(columns, results)))
        else:  # Apply to rows
            return self.process_dataframe_parallel(df, func, *args, **kwargs)

def parallel_groupby_apply(df: pd.DataFrame,
                          groupby_cols: List[str],
                          apply_func: Callable,
                          max_workers: int = None,
                          *args, **kwargs) -> pd.DataFrame:
    """
    Parallel groupby apply operation
    
    Args:
        df: Input DataFrame
        groupby_cols: Columns to group by
        apply_func: Function to apply to each group
        max_workers: Maximum number of workers
        *args, **kwargs: Arguments to pass to apply_func
    
    Returns:
        DataFrame with applied function
    """
    # Get unique groups
    groups = df.groupby(groupby_cols).groups
    
    # Process groups in parallel
    processor = ParallelProcessor(max_workers=max_workers)
    
    def process_group(group_key):
        group_df = df.loc[groups[group_key]]
        return apply_func(group_df, *args, **kwargs)
    
    with ProcessPoolExecutor(max_workers=max_workers or mp.cpu_count()) as executor:
        results = list(executor.map(process_group, groups.keys()))
    
    return pd.concat(results, ignore_index=True)

def parallel_rolling_apply(df: pd.DataFrame,
                          groupby_col: str,
                          window: int,
                          apply_func: Callable,
                          max_workers: int = None,
                          *args, **kwargs) -> pd.DataFrame:
    """
    Parallel rolling apply operation
    
    Args:
        df: Input DataFrame
        groupby_col: Column to group by
        window: Rolling window size
        apply_func: Function to apply
        max_workers: Maximum number of workers
        *args, **kwargs: Arguments to pass to apply_func
    
    Returns:
        DataFrame with rolling function applied
    """
    # Get unique groups
    groups = df[groupby_col].unique()
    
    def process_group(group_value):
        group_df = df[df[groupby_col] == group_value].copy()
        group_df = group_df.sort_values('quote_date')  # Assuming date column
        group_df['rolling_result'] = group_df.rolling(window=window).apply(apply_func, *args, **kwargs)
        return group_df
    
    with ProcessPoolExecutor(max_workers=max_workers or mp.cpu_count()) as executor:
        results = list(executor.map(process_group, groups))
    
    return pd.concat(results, ignore_index=True)

# Example usage and testing
if __name__ == "__main__":
    # Test parallel processor
    processor = ParallelProcessor(max_workers=4)
    
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C'], 10000),
        'value': np.random.randn(10000),
        'date': pd.date_range('2020-01-01', periods=10000, freq='D')
    })
    
    # Test parallel processing
    def test_func(df):
        return df.groupby('group')['value'].sum()
    
    start_time = time.time()
    result = processor.process_dataframe_parallel(sample_df, test_func)
    parallel_time = time.time() - start_time
    
    print(f"Parallel processing time: {parallel_time:.2f} seconds")
    print(f"Result shape: {result.shape}")

