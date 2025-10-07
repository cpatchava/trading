#!/usr/bin/env python3
"""
Trading Strategy Backtesting - Test options trading strategies
"""

import sqlite3
import pandas as pd
import numpy as np
from database_optimizations import OptimizedDBManager

def backtest_arbitrage_strategy(db_path: str, min_profit: float = 0.5, max_risk: float = 1.0):
    """
    Backtest a simple arbitrage strategy
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
            stock_price_contract_close,
            bid,
            ask,
            black_scholes_model_price,
            (bid + ask) / 2 as market_price,
            black_scholes_model_price - ((bid + ask) / 2) as price_difference
        FROM enriched_options 
        WHERE black_scholes_model_price IS NOT NULL 
            AND bid IS NOT NULL 
            AND ask IS NOT NULL 
            AND stock_price_today IS NOT NULL
            AND stock_price_contract_close IS NOT NULL
            AND ABS(black_scholes_model_price - ((bid + ask) / 2)) >= ?
        ORDER BY quote_date, underlying, expiration, strike
        """
        
        df = pd.read_sql_query(query, conn, params=(min_profit,), parse_dates=['quote_date', 'expiration'])
        return df

def calculate_strategy_performance(trades_df: pd.DataFrame):
    """
    Calculate performance metrics for the strategy
    """
    if trades_df.empty:
        return {}
    
    # Calculate P&L for each trade
    trades_df['theoretical_pnl'] = trades_df['price_difference']
    
    # Simple performance metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['theoretical_pnl'] > 0])
    losing_trades = len(trades_df[trades_df['theoretical_pnl'] < 0])
    
    total_pnl = trades_df['theoretical_pnl'].sum()
    avg_pnl = trades_df['theoretical_pnl'].mean()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Risk metrics
    max_profit = trades_df['theoretical_pnl'].max()
    max_loss = trades_df['theoretical_pnl'].min()
    std_pnl = trades_df['theoretical_pnl'].std()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'std_pnl': std_pnl,
        'sharpe_ratio': avg_pnl / std_pnl if std_pnl > 0 else 0
    }

def analyze_by_underlying(trades_df: pd.DataFrame):
    """
    Analyze performance by underlying symbol
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    performance_by_symbol = trades_df.groupby('underlying').agg({
        'theoretical_pnl': ['count', 'sum', 'mean', 'std'],
        'price_difference': ['min', 'max']
    }).round(4)
    
    performance_by_symbol.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'std_pnl', 'min_diff', 'max_diff']
    performance_by_symbol = performance_by_symbol.sort_values('total_pnl', ascending=False)
    
    return performance_by_symbol

def analyze_by_expiration(trades_df: pd.DataFrame):
    """
    Analyze performance by expiration date
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    # Add days to expiration
    trades_df['days_to_exp'] = (trades_df['expiration'] - trades_df['quote_date']).dt.days
    
    # Group by days to expiration ranges
    trades_df['exp_range'] = pd.cut(trades_df['days_to_exp'], 
                                   bins=[0, 7, 30, 90, 365, float('inf')], 
                                   labels=['<1 week', '1-4 weeks', '1-3 months', '3-12 months', '>1 year'])
    
    performance_by_exp = trades_df.groupby('exp_range').agg({
        'theoretical_pnl': ['count', 'sum', 'mean'],
        'price_difference': ['min', 'max']
    }).round(4)
    
    performance_by_exp.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'min_diff', 'max_diff']
    
    return performance_by_exp

def main():
    """Main backtesting function"""
    db_path = "../sql-database/options_data.db"
    
    print("🎯 Trading Strategy Backtesting")
    print("=" * 50)
    
    # Backtest arbitrage strategy
    print("\n📊 Backtesting arbitrage strategy (min $0.50 profit)...")
    trades_df = backtest_arbitrage_strategy(db_path, min_profit=0.5)
    
    if not trades_df.empty:
        print(f"   Found {len(trades_df):,} potential arbitrage trades")
        
        # Calculate performance
        performance = calculate_strategy_performance(trades_df)
        
        print(f"\n📈 Strategy Performance:")
        print(f"   Total trades: {performance['total_trades']:,}")
        print(f"   Winning trades: {performance['winning_trades']:,} ({performance['win_rate']:.1%})")
        print(f"   Losing trades: {performance['losing_trades']:,}")
        print(f"   Total P&L: ${performance['total_pnl']:,.2f}")
        print(f"   Average P&L per trade: ${performance['avg_pnl']:.2f}")
        print(f"   Maximum profit: ${performance['max_profit']:.2f}")
        print(f"   Maximum loss: ${performance['max_loss']:.2f}")
        print(f"   Sharpe ratio: {performance['sharpe_ratio']:.2f}")
        
        # Analyze by underlying
        print(f"\n🏆 Top 10 performing symbols:")
        symbol_performance = analyze_by_underlying(trades_df)
        print(symbol_performance.head(10).to_string())
        
        # Analyze by expiration
        print(f"\n📅 Performance by expiration range:")
        exp_performance = analyze_by_expiration(trades_df)
        print(exp_performance.to_string())
        
        # Recent opportunities
        print(f"\n🕒 Recent opportunities (last 10):")
        recent_trades = trades_df.tail(10)
        print(recent_trades[['underlying', 'strike', 'type', 'quote_date', 'expiration', 'price_difference']].to_string(index=False))
        
    else:
        print("❌ No arbitrage opportunities found with current criteria")
        print("   Try lowering the minimum profit threshold")

if __name__ == "__main__":
    main()

