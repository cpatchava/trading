#!/usr/bin/env python3
"""
Optimized price enrichment with vectorized operations and efficient lookups
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from database_optimizations import OptimizedDBManager, batch_insert_optimized
import time

def create_price_lookup_optimized(stocks_db: str) -> dict:
    """
    Create optimized price lookup dictionary with vectorized operations
    """
    with OptimizedDBManager(stocks_db).get_connection() as conn:
        # Use optimized query with proper indexing
        df = pd.read_sql_query(
            """
            SELECT symbol, quote_date, close 
            FROM stocks_enriched 
            WHERE close IS NOT NULL AND close > 0
            ORDER BY symbol, quote_date
            """,
            conn, 
            parse_dates=["quote_date"]
        )
    
    # Vectorized string formatting - much faster than individual formatting
    df["date_str"] = df["quote_date"].dt.strftime("%Y-%m-%d")
    
    # Create lookup dictionary using vectorized operations
    price_dict = dict(zip(
        zip(df["symbol"], df["date_str"]), 
        df["close"]
    ))
    
    print(f"🔑 Loaded {len(price_dict):,} prices into memory")
    return price_dict

def enrich_option_prices_optimized(
    options_db: str,
    stocks_db: str,
    chunk_size: int = 50000
):
    """Optimized price enrichment with vectorized operations"""
    
    # Initialize database managers
    options_db_manager = OptimizedDBManager(options_db)
    stocks_db_manager = OptimizedDBManager(stocks_db)
    
    # Create indexes for better performance
    print("🔧 Creating database indexes...")
    options_db_manager.create_indexes()
    stocks_db_manager.create_indexes()
    
    # Create optimized price lookup
    price_dict = create_price_lookup_optimized(stocks_db)
    
    # Add columns if needed
    with options_db_manager.get_connection() as conn:
        cur = conn.cursor()
        for col in ("stock_price_today REAL", "stock_price_contract_close REAL"):
            try:
                cur.execute(f"ALTER TABLE enriched_options ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        
        # Get total count for progress tracking
        total_count = cur.execute("SELECT COUNT(*) FROM enriched_options").fetchone()[0]
        print(f"🔢 Processing {total_count:,} rows in chunks of {chunk_size:,}")
    
    # Process in optimized chunks
    last_rowid = 0
    processed_count = 0
    
    while last_rowid < total_count:
        # Fetch chunk with optimized query
        with options_db_manager.get_connection() as conn:
            df_opt = pd.read_sql_query(
                """
                SELECT rowid, underlying, quote_date, expiration
                FROM enriched_options
                WHERE rowid > ?
                ORDER BY rowid
                LIMIT ?
                """,
                conn,
                params=(last_rowid, chunk_size),
                parse_dates=["quote_date", "expiration"]
            )
        
        if df_opt.empty:
            break
        
        # Vectorized date formatting
        df_opt["quote_date_str"] = df_opt["quote_date"].dt.strftime("%Y-%m-%d")
        df_opt["expiration_str"] = df_opt["expiration"].dt.strftime("%Y-%m-%d")
        
        # Vectorized lookup using pandas merge (much faster than list comprehension)
        # Create lookup DataFrame
        lookup_df = pd.DataFrame(list(price_dict.keys()), columns=['symbol', 'date'])
        lookup_df['price'] = list(price_dict.values())
        
        # Merge for quote date prices
        df_opt = df_opt.merge(
            lookup_df.rename(columns={'date': 'quote_date_str', 'price': 'stock_price_today'}),
            left_on=['underlying', 'quote_date_str'],
            right_on=['symbol', 'quote_date_str'],
            how='left'
        ).drop(columns=['symbol'])
        
        # Merge for expiration date prices
        df_opt = df_opt.merge(
            lookup_df.rename(columns={'date': 'expiration_str', 'price': 'stock_price_contract_close'}),
            left_on=['underlying', 'expiration_str'],
            right_on=['symbol', 'expiration_str'],
            how='left'
        ).drop(columns=['symbol'])
        
        # Clean up temporary columns
        df_opt = df_opt.drop(columns=['quote_date_str', 'expiration_str'])
        
        # Batch update with optimized connection
        updates = df_opt[['stock_price_today', 'stock_price_contract_close', 'rowid']].values.tolist()
        
        with options_db_manager.get_connection() as conn:
            conn.executemany(
                """
                UPDATE enriched_options 
                SET stock_price_today = ?, stock_price_contract_close = ? 
                WHERE rowid = ?
                """,
                updates
            )
            conn.commit()
        
        # Update progress
        processed_count += len(df_opt)
        progress = (processed_count / total_count) * 100
        last_rowid = int(df_opt["rowid"].max())
        
        print(f"📈 Progress: {progress:.1f}% - Updated rows {processed_count:,}/{total_count:,}")
    
    print("✅ Optimized price enrichment completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized price enrichment for options data"
    )
    parser.add_argument(
        "--options-db",
        default="../sql-database/options_data.db",
        help="Path to options SQLite DB"
    )
    parser.add_argument(
        "--stocks-db",
        default="../sql-database/stocks_data.db",
        help="Path to stocks_enriched SQLite DB"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of rows to process per batch"
    )
    args = parser.parse_args()
    
    start_time = time.time()
    enrich_option_prices_optimized(
        options_db=args.options_db,
        stocks_db=args.stocks_db,
        chunk_size=args.chunk_size
    )
    total_time = time.time() - start_time
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")

