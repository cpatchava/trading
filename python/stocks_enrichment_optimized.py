#!/usr/bin/env python3
"""
Optimized stocks enrichment with improved performance and memory efficiency
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from database_optimizations import OptimizedDBManager, batch_insert_optimized
import time

# Time windows for volatility (in trading days)
WINDOWS = {
    'volatility_1_week': 5,
    'volatility_2_weeks': 10,
    'volatility_1_month': 21,
    'volatility_3_months': 63
}

def load_sp500_symbols(path: str) -> set:
    """Load S&P 500 symbols from a text file, one per line."""
    with open(path, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def calculate_volatility_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized volatility calculation with optimized operations
    """
    # Sort by symbol and date for rolling calculations
    df = df.sort_values(['symbol', 'quote_date']).copy()
    
    # Vectorized log returns calculation
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Handle infinite values efficiently
    df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling volatility using vectorized operations
    for col, window in WINDOWS.items():
        # Use groupby with transform for vectorized rolling calculations
        df[col] = (
            df.groupby('symbol')['log_return']
              .transform(lambda x: x.rolling(window, min_periods=1).std(ddof=0) * np.sqrt(252))
        )
    
    # Drop intermediate column
    df = df.drop(columns=['log_return'])
    
    return df

def enrich_stocks_table_optimized(db_path: str, sp500_symbols: set, chunk_size: int = 100000):
    """
    Optimized stocks enrichment with chunked processing and memory efficiency
    """
    db_manager = OptimizedDBManager(db_path)
    
    # Create indexes first
    print("🔧 Creating database indexes...")
    db_manager.create_indexes()
    
    # Load data in chunks to manage memory usage
    print("📊 Loading stocks data...")
    
    with db_manager.get_connection() as conn:
        # Get total count
        total_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        print(f"📈 Total rows to process: {total_count:,}")
        
        # Process in chunks
        processed_count = 0
        first_chunk = True
        
        # Use optimized chunked reading
        chunk_reader = pd.read_sql_query(
            "SELECT * FROM stocks ORDER BY symbol, quote_date",
            conn,
            chunksize=chunk_size,
            parse_dates=['quote_date']
        )
        
        for chunk_df in chunk_reader:
            # Filter out invalid data efficiently
            chunk_df = chunk_df.dropna(subset=['quote_date', 'close']).copy()
            chunk_df = chunk_df[chunk_df['close'] > 0]
            
            if chunk_df.empty:
                continue
            
            # Flag S&P 500 membership using vectorized operation
            chunk_df['is_sp500'] = chunk_df['symbol'].isin(sp500_symbols)
            
            # Calculate volatility using optimized function
            chunk_df = calculate_volatility_vectorized(chunk_df)
            
            # Write chunk to database
            with db_manager.get_connection() as write_conn:
                if first_chunk:
                    # Drop existing table and create new one
                    write_conn.execute('DROP TABLE IF EXISTS stocks_enriched')
                    write_conn.commit()
                    first_chunk = False
                
                # Use optimized batch insert
                batch_insert_optimized(
                    write_conn, 
                    'stocks_enriched', 
                    chunk_df,
                    batch_size=10000
                )
            
            # Update progress
            processed_count += len(chunk_df)
            progress = (processed_count / total_count) * 100
            print(f"📈 Progress: {progress:.1f}% - Processed {processed_count:,}/{total_count:,} rows")
    
    # Analyze table for query optimization
    print("🔍 Analyzing table for query optimization...")
    db_manager.analyze_tables()
    
    print("✅ Optimized stocks enrichment completed!")

def main():
    """Main function with optimized processing"""
    # Paths
    db_path = os.path.join('..', 'sql-database', 'stocks_data.db')
    sp500_file = os.path.join('..', 'sql-database', 'sp500_symbols.txt')
    
    # Ensure DB directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Load S&P 500 symbols
    syms = load_sp500_symbols(sp500_file)
    print(f"📋 Loaded {len(syms)} S&P 500 symbols")
    
    # Enrich stocks data with optimized processing
    start_time = time.time()
    enrich_stocks_table_optimized(db_path, syms)
    total_time = time.time() - start_time
    
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")
    print("✅ 'stocks_enriched' table created with SP500 flag and volatility metrics")

if __name__ == '__main__':
    main()

