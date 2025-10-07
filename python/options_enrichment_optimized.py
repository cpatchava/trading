#!/usr/bin/env python3
"""
Optimized options enrichment with improved performance
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from database_optimizations import OptimizedDBManager, batch_insert_optimized
import time

def process_chunk_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized worker: vectorized datetime operations and efficient data processing
    """
    # Vectorized datetime conversion - much faster than pd.to_datetime on individual columns
    df = df.copy()  # Avoid SettingWithCopyWarning
    
    # Use vectorized operations for datetime conversion
    df["expiration"] = pd.to_datetime(df["expiration"], errors='coerce')
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors='coerce')
    
    # Vectorized days calculation
    df["num_days"] = (df["expiration"] - df["quote_date"]).dt.days
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['expiration', 'quote_date'])
    
    return df

def main_optimized(db_path: str, chunk_size: int, workers: int):
    """Optimized main function with better memory management and performance"""
    
    # Initialize optimized database manager
    db_manager = OptimizedDBManager(db_path)
    
    # Create indexes first for better performance
    print("🔧 Creating database indexes...")
    db_manager.create_indexes()
    
    # Drop existing table
    with db_manager.get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS enriched_options")
        conn.commit()
    
    # Use optimized connection for reading
    with db_manager.get_connection() as conn:
        # Get total count for progress tracking
        total_count = conn.execute("SELECT COUNT(*) FROM options").fetchone()[0]
        print(f"📊 Total rows to process: {total_count:,}")
    
    # Process in chunks with optimized memory usage
    last_rowid = 0
    first = True
    processed_count = 0
    
    # Pre-allocate pool for better performance
    with Pool(workers) as pool:
        while True:
            # Fetch chunk with optimized query
            with db_manager.get_connection() as reader_conn:
                df_chunk = pd.read_sql_query(
                    """
                    SELECT rowid AS rowid, *
                    FROM options
                    WHERE rowid > ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    reader_conn,
                    params=(last_rowid, chunk_size),
                    parse_dates=['expiration', 'quote_date']  # Parse dates during read
                )
            
            if df_chunk.empty:
                break
            
            # Update progress
            processed_count += len(df_chunk)
            progress = (processed_count / total_count) * 100
            print(f"📈 Progress: {progress:.1f}% ({processed_count:,}/{total_count:,})")
            
            # Advance pointer
            last_rowid = int(df_chunk["rowid"].max())
            
            # Drop helper column
            df_chunk = df_chunk.drop(columns=["rowid"])
            
            # Split and process in parallel with optimized function
            sub_dfs = np.array_split(df_chunk, workers)
            processed_list = pool.map(process_chunk_optimized, sub_dfs)
            
            # Recombine efficiently
            df_enriched = pd.concat(processed_list, ignore_index=True)
            
            # Write with optimized batch insert
            with db_manager.get_connection() as writer_conn:
                batch_insert_optimized(
                    writer_conn, 
                    "enriched_options", 
                    df_enriched,
                    batch_size=10000
                )
            
            first = False
    
    # Analyze table for query optimization
    print("🔍 Analyzing table for query optimization...")
    db_manager.analyze_tables()
    
    print("✅ Optimized enrichment completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized chunked & parallel options enrichment"
    )
    parser.add_argument(
        "--options-db",
        default="../sql-database/options_data.db",
        help="Path to your options SQLite DB"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows to fetch & process per batch"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel worker processes"
    )
    args = parser.parse_args()
    
    start_time = time.time()
    main_optimized(
        db_path=args.options_db,
        chunk_size=args.chunk_size,
        workers=args.workers
    )
    total_time = time.time() - start_time
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")

