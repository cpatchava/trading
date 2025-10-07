#!/usr/bin/env python3
"""
Volatility Analysis - Analyze market volatility patterns
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from database_optimizations import OptimizedDBManager

def analyze_market_volatility(db_path: str):
    """
    Analyze overall market volatility patterns
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        # Get volatility data from stocks_enriched
        query = """
        SELECT 
            symbol,
            quote_date,
            close,
            volatility_1_week,
            volatility_2_weeks,
            volatility_1_month,
            volatility_3_months,
            is_sp500
        FROM stocks_enriched 
        WHERE volatility_1_month IS NOT NULL
        ORDER BY quote_date, symbol
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['quote_date'])
        return df

def find_volatility_opportunities(db_path: str):
    """
    Find stocks with unusual volatility patterns
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        query = """
        SELECT 
            symbol,
            quote_date,
            close,
            volatility_1_week,
            volatility_1_month,
            volatility_3_months,
            is_sp500,
            (volatility_1_week - volatility_1_month) as vol_spread_short,
            (volatility_1_month - volatility_3_months) as vol_spread_long
        FROM stocks_enriched 
        WHERE volatility_1_week IS NOT NULL 
            AND volatility_1_month IS NOT NULL 
            AND volatility_3_months IS NOT NULL
        ORDER BY quote_date DESC, symbol
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['quote_date'])
        return df

def analyze_sp500_volatility(db_path: str):
    """
    Analyze S&P 500 volatility specifically
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        query = """
        SELECT 
            quote_date,
            AVG(volatility_1_week) as avg_vol_1w,
            AVG(volatility_1_month) as avg_vol_1m,
            AVG(volatility_3_months) as avg_vol_3m,
            COUNT(*) as stock_count
        FROM stocks_enriched 
        WHERE is_sp500 = 1 
            AND volatility_1_month IS NOT NULL
        GROUP BY quote_date
        ORDER BY quote_date
        """
        
        df = pd.read_sql_query(query, conn, parse_dates=['quote_date'])
        return df

def main():
    """Main volatility analysis function"""
    db_path = "../sql-database/stocks_data.db"
    
    print("📈 Volatility Analysis")
    print("=" * 50)
    
    # Analyze market volatility
    print("\n📊 Loading market volatility data...")
    vol_df = analyze_market_volatility(db_path)
    
    if not vol_df.empty:
        print(f"   Loaded {len(vol_df):,} volatility records")
        print(f"   Date range: {vol_df['quote_date'].min()} to {vol_df['quote_date'].max()}")
        print(f"   Unique symbols: {vol_df['symbol'].nunique():,}")
        print(f"   S&P 500 stocks: {vol_df[vol_df['is_sp500']==1]['symbol'].nunique():,}")
        
        # Current volatility statistics
        latest_date = vol_df['quote_date'].max()
        latest_vol = vol_df[vol_df['quote_date'] == latest_date]
        
        print(f"\n📊 Latest volatility (as of {latest_date.date()}):")
        print(f"   Average 1-month volatility: {latest_vol['volatility_1_month'].mean():.2%}")
        print(f"   Average 3-month volatility: {latest_vol['volatility_3_months'].mean():.2%}")
        print(f"   Highest 1-month volatility: {latest_vol['volatility_1_month'].max():.2%}")
        print(f"   Lowest 1-month volatility: {latest_vol['volatility_1_month'].min():.2%}")
    
    # Find volatility opportunities
    print(f"\n🔍 Finding volatility opportunities...")
    vol_opps = find_volatility_opportunities(db_path)
    
    if not vol_opps.empty:
        # Find stocks with high volatility spreads
        high_spread = vol_opps[vol_opps['vol_spread_short'] > 0.1]  # 10% spread
        print(f"   Found {len(high_spread)} stocks with high short-term volatility spreads")
        
        if len(high_spread) > 0:
            print(f"\n🏆 Top 10 stocks with highest volatility spreads:")
            top_spreads = high_spread.nlargest(10, 'vol_spread_short')
            print(top_spreads[['symbol', 'quote_date', 'volatility_1_week', 'volatility_1_month', 'vol_spread_short']].to_string(index=False))
    
    # S&P 500 volatility analysis
    print(f"\n📊 S&P 500 Volatility Analysis...")
    sp500_vol = analyze_sp500_volatility(db_path)
    
    if not sp500_vol.empty:
        print(f"   S&P 500 volatility trends over {len(sp500_vol)} trading days")
        print(f"   Average 1-month volatility: {sp500_vol['avg_vol_1m'].mean():.2%}")
        print(f"   Average 3-month volatility: {sp500_vol['avg_vol_3m'].mean():.2%}")
        
        # Find high volatility periods
        high_vol_periods = sp500_vol[sp500_vol['avg_vol_1m'] > sp500_vol['avg_vol_1m'].quantile(0.9)]
        print(f"   High volatility periods (>90th percentile): {len(high_vol_periods)} days")
        
        if len(high_vol_periods) > 0:
            print(f"   Recent high volatility dates:")
            recent_high = high_vol_periods.tail(5)
            print(recent_high[['quote_date', 'avg_vol_1m', 'avg_vol_3m']].to_string(index=False))

if __name__ == "__main__":
    main()

