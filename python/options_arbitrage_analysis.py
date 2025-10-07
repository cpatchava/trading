#!/usr/bin/env python3
"""
Options Arbitrage Analysis - Find mispriced options
"""

import sqlite3
import pandas as pd
import numpy as np
from database_optimizations import OptimizedDBManager

def find_arbitrage_opportunities(db_path: str, min_price_diff: float = 0.5, limit: int = 100):
    """
    Find options with significant price differences between market and theoretical prices
    
    Args:
        db_path: Path to options database
        min_price_diff: Minimum price difference to consider (in dollars)
        limit: Maximum number of results to return
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
            ABS(black_scholes_model_price - ((bid + ask) / 2)) as abs_price_difference,
            (black_scholes_model_price - ((bid + ask) / 2)) / ((bid + ask) / 2) * 100 as percent_difference
        FROM enriched_options 
        WHERE black_scholes_model_price IS NOT NULL 
            AND bid IS NOT NULL 
            AND ask IS NOT NULL 
            AND stock_price_today IS NOT NULL
            AND ABS(black_scholes_model_price - ((bid + ask) / 2)) >= ?
        ORDER BY ABS(black_scholes_model_price - ((bid + ask) / 2)) DESC
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(min_price_diff, limit))
        return df

def analyze_volatility_surface(db_path: str, symbol: str = None):
    """
    Analyze volatility surface for a specific symbol or overall market
    """
    with OptimizedDBManager(db_path).get_connection() as conn:
        if symbol:
            where_clause = "WHERE underlying = ?"
            params = (symbol,)
        else:
            where_clause = ""
            params = ()
            
        query = f"""
        SELECT 
            underlying,
            strike,
            type,
            expiration,
            quote_date,
            stock_price_today,
            black_scholes_model_price,
            (bid + ask) / 2 as market_price,
            ABS(black_scholes_model_price - ((bid + ask) / 2)) as price_difference
        FROM enriched_options 
        {where_clause}
        AND black_scholes_model_price IS NOT NULL 
        AND bid IS NOT NULL 
        AND ask IS NOT NULL 
        AND stock_price_today IS NOT NULL
        ORDER BY underlying, expiration, strike
        """
        
        df = pd.read_sql_query(query, conn, params=params)
        return df

def main():
    """Main analysis function"""
    db_path = "../sql-database/options_data.db"
    
    print("🔍 Options Arbitrage Analysis")
    print("=" * 50)
    
    # Find arbitrage opportunities
    print("\n📊 Finding arbitrage opportunities (min $0.50 difference)...")
    arbitrage_df = find_arbitrage_opportunities(db_path, min_price_diff=0.5, limit=20)
    
    if not arbitrage_df.empty:
        print(f"\n🎯 Found {len(arbitrage_df)} potential arbitrage opportunities:")
        print(arbitrage_df[['underlying', 'strike', 'type', 'expiration', 'market_price', 'black_scholes_model_price', 'price_difference', 'percent_difference']].to_string(index=False))
        
        # Summary statistics
        print(f"\n📈 Arbitrage Summary:")
        print(f"   Average price difference: ${arbitrage_df['abs_price_difference'].mean():.2f}")
        print(f"   Maximum price difference: ${arbitrage_df['abs_price_difference'].max():.2f}")
        print(f"   Average percent difference: {arbitrage_df['percent_difference'].abs().mean():.1f}%")
        
        # Top opportunities by percentage
        print(f"\n🏆 Top 5 opportunities by percentage:")
        top_percent = arbitrage_df.nlargest(5, 'percent_difference')
        print(top_percent[['underlying', 'strike', 'type', 'percent_difference']].to_string(index=False))
    else:
        print("❌ No significant arbitrage opportunities found with current criteria")
    
    # Analyze specific symbol (example: AAPL)
    print(f"\n📊 Analyzing volatility surface for AAPL...")
    aapl_df = analyze_volatility_surface(db_path, "AAPL")
    
    if not aapl_df.empty:
        print(f"   Found {len(aapl_df)} AAPL options")
        print(f"   Average price difference: ${aapl_df['price_difference'].abs().mean():.2f}")
        print(f"   Date range: {aapl_df['quote_date'].min()} to {aapl_df['quote_date'].max()}")
    else:
        print("   No AAPL options found in database")

if __name__ == "__main__":
    main()

