#!/usr/bin/env python3
"""
Data Quality Analysis - Identify and fix data quality issues
"""

import sqlite3
import pandas as pd
import numpy as np
from database_optimizations import OptimizedDBManager

def analyze_data_quality_issues(db_path: str):
    """
    Analyze data quality issues in the options database
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        # Check for extreme price differences
        query = """
        SELECT 
            underlying,
            strike,
            type,
            expiration,
            quote_date,
            stock_price_today,
            bid,
            ask,
            black_scholes_model_price,
            (bid + ask) / 2 as market_price,
            black_scholes_model_price - ((bid + ask) / 2) as price_difference,
            ABS(black_scholes_model_price - ((bid + ask) / 2)) as abs_price_difference
        FROM enriched_options 
        WHERE black_scholes_model_price IS NOT NULL 
            AND bid IS NOT NULL 
            AND ask IS NOT NULL 
            AND stock_price_today IS NOT NULL
        ORDER BY ABS(black_scholes_model_price - ((bid + ask) / 2)) DESC
        LIMIT 100
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['quote_date', 'expiration'])
        return df

def identify_problematic_data(df: pd.DataFrame):
    """
    Identify specific data quality issues
    """
    issues = {}
    
    # Check for extreme Black-Scholes prices
    extreme_bs = df[df['black_scholes_model_price'] > 10000]
    if not extreme_bs.empty:
        issues['extreme_bs_prices'] = {
            'count': len(extreme_bs),
            'symbols': extreme_bs['underlying'].unique().tolist(),
            'max_price': extreme_bs['black_scholes_model_price'].max()
        }
    
    # Check for zero or negative market prices
    zero_market = df[(df['bid'] <= 0) | (df['ask'] <= 0)]
    if not zero_market.empty:
        issues['zero_market_prices'] = {
            'count': len(zero_market),
            'symbols': zero_market['underlying'].unique().tolist()
        }
    
    # Check for missing volatility data
    missing_vol = df[df['black_scholes_model_price'].isna()]
    if not missing_vol.empty:
        issues['missing_bs_prices'] = {
            'count': len(missing_vol)
        }
    
    # Check for unrealistic stock prices
    extreme_stock = df[(df['stock_price_today'] > 10000) | (df['stock_price_today'] < 0.01)]
    if not extreme_stock.empty:
        issues['extreme_stock_prices'] = {
            'count': len(extreme_stock),
            'symbols': extreme_stock['underlying'].unique().tolist(),
            'price_range': (extreme_stock['stock_price_today'].min(), extreme_stock['stock_price_today'].max())
        }
    
    return issues

def get_clean_data_sample(db_path: str, limit: int = 1000):
    """
    Get a sample of clean, realistic data for analysis
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        query = """
        SELECT 
            underlying,
            strike,
            type,
            expiration,
            quote_date,
            stock_price_today,
            bid,
            ask,
            black_scholes_model_price,
            (bid + ask) / 2 as market_price,
            black_scholes_model_price - ((bid + ask) / 2) as price_difference,
            ABS(black_scholes_model_price - ((bid + ask) / 2)) as abs_price_difference
        FROM enriched_options 
        WHERE black_scholes_model_price IS NOT NULL 
            AND bid IS NOT NULL 
            AND ask IS NOT NULL 
            AND stock_price_today IS NOT NULL
            AND black_scholes_model_price BETWEEN 0.01 AND 1000
            AND bid > 0 AND ask > 0
            AND stock_price_today BETWEEN 1 AND 1000
            AND ABS(black_scholes_model_price - ((bid + ask) / 2)) < 50
        ORDER BY quote_date DESC, underlying
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,), parse_dates=['quote_date', 'expiration'])
        return df

def analyze_put_options_quality(db_path: str):
    """
    Specifically analyze put options data quality for our strategy
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        query = """
        SELECT 
            underlying,
            strike,
            type,
            expiration,
            quote_date,
            stock_price_today,
            bid,
            ask,
            black_scholes_model_price,
            (bid + ask) / 2 as market_price,
            (expiration - quote_date) as days_to_expiration,
            ABS(black_scholes_model_price - ((bid + ask) / 2)) as abs_price_difference
        FROM enriched_options 
        WHERE type = 'put'
            AND black_scholes_model_price IS NOT NULL 
            AND bid IS NOT NULL 
            AND ask IS NOT NULL 
            AND stock_price_today IS NOT NULL
            AND (expiration - quote_date) <= 30  -- Less than 1 month
            AND black_scholes_model_price BETWEEN 0.01 AND 100
            AND bid > 0 AND ask > 0
            AND stock_price_today BETWEEN 1 AND 1000
        ORDER BY quote_date DESC, underlying
        LIMIT 1000
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['quote_date', 'expiration'])
        return df

def main():
    """Main data quality analysis function"""
    db_path = "../sql-database/options_data.db"
    
    print("🔍 Data Quality Analysis")
    print("=" * 50)
    
    # Analyze data quality issues
    print("\n📊 Analyzing data quality issues...")
    problematic_df = analyze_data_quality_issues(db_path)
    
    if not problematic_df.empty:
        print(f"   Found {len(problematic_df)} records with extreme price differences")
        
        # Identify specific issues
        issues = identify_problematic_data(problematic_df)
        
        print(f"\n🚨 Data Quality Issues Found:")
        for issue_type, details in issues.items():
            print(f"   {issue_type}: {details}")
        
        # Show examples of problematic data
        print(f"\n📋 Examples of problematic data:")
        print(problematic_df.head(10)[['underlying', 'strike', 'type', 'market_price', 'black_scholes_model_price', 'abs_price_difference']].to_string(index=False))
    
    # Get clean data sample
    print(f"\n✅ Getting clean data sample...")
    clean_df = get_clean_data_sample(db_path, limit=1000)
    
    if not clean_df.empty:
        print(f"   Found {len(clean_df)} clean records")
        print(f"   Date range: {clean_df['quote_date'].min()} to {clean_df['quote_date'].max()}")
        print(f"   Unique symbols: {clean_df['underlying'].nunique()}")
        print(f"   Average price difference: ${clean_df['abs_price_difference'].mean():.2f}")
        print(f"   Max price difference: ${clean_df['abs_price_difference'].max():.2f}")
        
        # Show sample of clean data
        print(f"\n📋 Sample of clean data:")
        print(clean_df.head(10)[['underlying', 'strike', 'type', 'market_price', 'black_scholes_model_price', 'abs_price_difference']].to_string(index=False))
    
    # Analyze put options specifically
    print(f"\n📊 Analyzing put options data quality...")
    puts_df = analyze_put_options_quality(db_path)
    
    if not puts_df.empty:
        print(f"   Found {len(puts_df)} clean put options (< 1 month)")
        print(f"   Average days to expiration: {puts_df['days_to_expiration'].mean():.1f}")
        print(f"   Strike price range: ${puts_df['strike'].min():.2f} - ${puts_df['strike'].max():.2f}")
        print(f"   Stock price range: ${puts_df['stock_price_today'].min():.2f} - ${puts_df['stock_price_today'].max():.2f}")
        
        # Show sample put options
        print(f"\n📋 Sample put options:")
        print(puts_df.head(10)[['underlying', 'strike', 'expiration', 'days_to_expiration', 'market_price', 'black_scholes_model_price']].to_string(index=False))
    else:
        print("   ❌ No clean put options found with current criteria")

if __name__ == "__main__":
    main()
