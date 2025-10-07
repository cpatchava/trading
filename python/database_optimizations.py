#!/usr/bin/env python3
"""
Database optimization utilities for improved performance
"""

import sqlite3
import pandas as pd
from contextlib import contextmanager
from typing import Generator, Optional

class OptimizedDBManager:
    """Optimized database connection manager with connection pooling and performance settings"""
    
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = []
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Apply SQLite performance optimizations"""
        conn = sqlite3.connect(self.db_path)
        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Balanced safety/performance
        conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
        conn.execute("PRAGMA optimize")  # Optimize query planner
        conn.commit()
        conn.close()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with automatic cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Apply performance settings to each connection
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            yield conn
        finally:
            conn.close()
    
    def create_indexes(self):
        """Create essential indexes for better query performance"""
        with self.get_connection() as conn:
            indexes = [
                # Options table indexes
                "CREATE INDEX IF NOT EXISTS idx_options_underlying ON options(underlying)",
                "CREATE INDEX IF NOT EXISTS idx_options_quote_date ON options(quote_date)",
                "CREATE INDEX IF NOT EXISTS idx_options_expiration ON options(expiration)",
                "CREATE INDEX IF NOT EXISTS idx_options_underlying_date ON options(underlying, quote_date)",
                
                # Stocks table indexes
                "CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_quote_date ON stocks(quote_date)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_symbol_date ON stocks(symbol, quote_date)",
                
                # Enriched tables indexes
                "CREATE INDEX IF NOT EXISTS idx_enriched_options_underlying ON enriched_options(underlying)",
                "CREATE INDEX IF NOT EXISTS idx_enriched_options_quote_date ON enriched_options(quote_date)",
                "CREATE INDEX IF NOT EXISTS idx_enriched_options_rowid ON enriched_options(rowid)",
                "CREATE INDEX IF NOT EXISTS idx_enriched_options_pricing ON enriched_options(black_scholes_model_price)",
                
                "CREATE INDEX IF NOT EXISTS idx_stocks_enriched_symbol ON stocks_enriched(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_enriched_quote_date ON stocks_enriched(quote_date)",
                "CREATE INDEX IF NOT EXISTS idx_stocks_enriched_symbol_date ON stocks_enriched(symbol, quote_date)",
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                    print(f"✅ Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                except sqlite3.Error as e:
                    print(f"⚠️ Index creation failed: {e}")
            
            conn.commit()
    
    def analyze_tables(self):
        """Analyze tables for query optimization"""
        with self.get_connection() as conn:
            tables = ['options', 'stocks', 'enriched_options', 'stocks_enriched']
            for table in tables:
                try:
                    result = conn.execute(f"ANALYZE {table}")
                    print(f"✅ Analyzed table: {table}")
                except sqlite3.Error as e:
                    print(f"⚠️ Analysis failed for {table}: {e}")

def batch_insert_optimized(conn: sqlite3.Connection, table_name: str, 
                          df: pd.DataFrame, batch_size: int = 10000):
    """Optimized batch insert with transaction management"""
    if df.empty:
        return
    
    # Prepare data for insertion
    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    
    # Process in batches
    total_rows = len(df)
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Convert to list of tuples for executemany
        data_tuples = [tuple(row) for row in batch_df.values]
        
        conn.executemany(insert_sql, data_tuples)
        conn.commit()
        
        print(f"↳ Inserted rows {start_idx+1}-{end_idx} of {total_rows}")

def optimized_read_sql_chunked(conn: sqlite3.Connection, query: str, 
                              chunk_size: int = 50000, **kwargs) -> Generator[pd.DataFrame, None, None]:
    """Memory-efficient chunked reading with optimized parameters"""
    return pd.read_sql_query(
        query, 
        conn, 
        chunksize=chunk_size,
        parse_dates=kwargs.get('parse_dates', []),
        dtype=kwargs.get('dtype', None)
    )

if __name__ == "__main__":
    # Example usage
    db_manager = OptimizedDBManager("../sql-database/options_data.db")
    db_manager.create_indexes()
    db_manager.analyze_tables()

