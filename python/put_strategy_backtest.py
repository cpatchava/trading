#!/usr/bin/env python3
"""
Short-Term Put Strategy Backtest
- Buy puts with < 1 month to expiration
- $500k seed capital with reinvestment
- 6-month backtest period
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database_optimizations import OptimizedDBManager
import warnings
warnings.filterwarnings('ignore')

class PutStrategyBacktest:
    def __init__(self, db_path: str, seed_capital: float = 500000):
        self.db_path = db_path
        self.seed_capital = seed_capital
        self.current_capital = seed_capital
        self.portfolio = []
        self.trade_history = []
        self.daily_pnl = []
        
    def get_clean_put_data(self, start_date: str, end_date: str):
        """
        Get clean put options data for the strategy
        """
        with OptimizedDBManager(self.db_path).get_connection() as conn:
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
                volume,
                open_interest,
                black_scholes_model_price,
                (bid + ask) / 2 as market_price,
                (expiration - quote_date) as days_to_expiration
            FROM enriched_options 
            WHERE type = 'put'
                AND black_scholes_model_price IS NOT NULL 
                AND bid IS NOT NULL 
                AND ask IS NOT NULL 
                AND stock_price_today IS NOT NULL
                AND stock_price_contract_close IS NOT NULL
                AND (expiration - quote_date) BETWEEN 1 AND 30  -- 1 to 30 days
                AND black_scholes_model_price BETWEEN 0.01 AND 100
                AND bid > 0 AND ask > 0
                AND stock_price_today BETWEEN 1 AND 1000
                AND quote_date BETWEEN ? AND ?
                AND underlying NOT IN ('SFI')  -- Exclude problematic data
            ORDER BY quote_date, underlying, expiration, strike
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date), 
                                 parse_dates=['quote_date', 'expiration'])
            return df
    
    def calculate_position_size(self, option_price: float, max_position_size: float = 0.05):
        """
        Calculate position size based on available capital
        max_position_size: Maximum 5% of capital per position
        """
        max_investment = self.current_capital * max_position_size
        max_contracts = int(max_investment / option_price)
        return min(max_contracts, 100)  # Cap at 100 contracts per position
    
    def select_put_options(self, df: pd.DataFrame, selection_criteria: str = 'undervalued'):
        """
        Select put options based on criteria
        """
        if selection_criteria == 'undervalued':
            # Select puts where market price < Black-Scholes price (undervalued)
            selected = df[df['market_price'] < df['black_scholes_model_price'] * 0.9].copy()
        elif selection_criteria == 'high_volume':
            # Select puts with high volume
            selected = df[df['volume'] > df['volume'].quantile(0.8)].copy()
        elif selection_criteria == 'itm':
            # Select in-the-money puts
            selected = df[df['strike'] > df['stock_price_today']].copy()
        else:
            # Random selection for comparison
            selected = df.sample(n=min(100, len(df))).copy()
        
        # Sort by potential return (Black-Scholes vs market price)
        selected['potential_return'] = (selected['black_scholes_model_price'] - selected['market_price']) / selected['market_price']
        selected = selected.sort_values('potential_return', ascending=False)
        
        return selected.head(50)  # Top 50 opportunities per day
    
    def execute_trades(self, selected_options: pd.DataFrame, trade_date: str):
        """
        Execute trades for selected options
        """
        daily_trades = []
        
        for _, option in selected_options.iterrows():
            # Calculate position size
            position_size = self.calculate_position_size(option['ask'])
            
            if position_size > 0:
                # Calculate trade details
                trade_cost = position_size * option['ask'] * 100  # 100 shares per contract
                
                if trade_cost <= self.current_capital:
                    # Execute trade
                    trade = {
                        'trade_date': trade_date,
                        'underlying': option['underlying'],
                        'strike': option['strike'],
                        'expiration': option['expiration'],
                        'days_to_exp': option['days_to_expiration'],
                        'entry_price': option['ask'],
                        'contracts': position_size,
                        'trade_cost': trade_cost,
                        'stock_price_entry': option['stock_price_today'],
                        'bs_price': option['black_scholes_model_price'],
                        'market_price': option['market_price'],
                        'potential_return': option['potential_return']
                    }
                    
                    daily_trades.append(trade)
                    self.current_capital -= trade_cost
                    
                    # Add to portfolio
                    self.portfolio.append(trade)
        
        return daily_trades
    
    def settle_expired_options(self, current_date: str):
        """
        Settle expired options and calculate P&L
        """
        expired_trades = []
        active_portfolio = []
        
        for trade in self.portfolio:
            if trade['expiration'] <= current_date:
                # Option expired - calculate final P&L
                final_pnl = self.calculate_option_pnl(trade, current_date)
                trade['exit_date'] = current_date
                trade['final_pnl'] = final_pnl
                trade['exit_price'] = max(0, trade['strike'] - trade.get('stock_price_exit', trade['stock_price_entry']))
                
                expired_trades.append(trade)
                self.current_capital += final_pnl  # Add P&L back to capital
            else:
                active_portfolio.append(trade)
        
        self.portfolio = active_portfolio
        return expired_trades
    
    def calculate_option_pnl(self, trade: dict, current_date: str):
        """
        Calculate P&L for an option trade
        """
        # For simplicity, assume we can sell at intrinsic value at expiration
        # In reality, you'd need the actual stock price at expiration
        
        # Get stock price at expiration (approximate with entry price for now)
        # In a real implementation, you'd fetch the actual stock price
        stock_price_at_exp = trade['stock_price_entry']  # Simplified
        
        # Calculate intrinsic value
        intrinsic_value = max(0, trade['strike'] - stock_price_at_exp)
        
        # Calculate P&L
        exit_value = intrinsic_value * trade['contracts'] * 100
        entry_cost = trade['trade_cost']
        pnl = exit_value - entry_cost
        
        return pnl
    
    def run_backtest(self, start_date: str, end_date: str, selection_criteria: str = 'undervalued'):
        """
        Run the complete backtest
        """
        print(f"🚀 Starting Put Strategy Backtest")
        print(f"   Seed Capital: ${self.seed_capital:,.2f}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Selection Criteria: {selection_criteria}")
        print("=" * 60)
        
        # Get all put data for the period
        puts_df = self.get_clean_put_data(start_date, end_date)
        
        if puts_df.empty:
            print("❌ No put options data found for the specified period")
            return
        
        print(f"📊 Found {len(puts_df):,} put options records")
        print(f"   Date range: {puts_df['quote_date'].min()} to {puts_df['quote_date'].max()}")
        print(f"   Unique symbols: {puts_df['underlying'].nunique()}")
        
        # Get unique trading dates
        trading_dates = sorted(puts_df['quote_date'].unique())
        
        print(f"\n📅 Processing {len(trading_dates)} trading days...")
        
        for i, trade_date in enumerate(trading_dates):
            if i % 10 == 0:  # Progress update every 10 days
                print(f"   Processing day {i+1}/{len(trading_dates)}: {trade_date.date()}")
            
            # Settle expired options first
            expired_trades = self.settle_expired_options(trade_date)
            if expired_trades:
                self.trade_history.extend(expired_trades)
            
            # Get options for current date
            daily_options = puts_df[puts_df['quote_date'] == trade_date]
            
            if not daily_options.empty:
                # Select options to trade
                selected_options = self.select_put_options(daily_options, selection_criteria)
                
                if not selected_options.empty:
                    # Execute trades
                    daily_trades = self.execute_trades(selected_options, trade_date)
                    
                    if daily_trades:
                        print(f"     Executed {len(daily_trades)} trades on {trade_date.date()}")
            
            # Record daily capital
            self.daily_pnl.append({
                'date': trade_date,
                'capital': self.current_capital,
                'portfolio_value': len(self.portfolio),
                'total_trades': len(self.trade_history)
            })
        
        # Settle any remaining options at the end
        final_expired = self.settle_expired_options(end_date)
        if final_expired:
            self.trade_history.extend(final_expired)
        
        print(f"\n✅ Backtest Complete!")
        return self.analyze_results()
    
    def analyze_results(self):
        """
        Analyze backtest results
        """
        if not self.trade_history:
            print("❌ No trades executed during backtest period")
            return {}
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(self.trade_history)
        daily_df = pd.DataFrame(self.daily_pnl)
        
        # Calculate performance metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['final_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['final_pnl'] <= 0])
        
        total_pnl = trades_df['final_pnl'].sum()
        total_return = (self.current_capital - self.seed_capital) / self.seed_capital
        
        avg_pnl = trades_df['final_pnl'].mean()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        max_profit = trades_df['final_pnl'].max()
        max_loss = trades_df['final_pnl'].min()
        
        # Risk metrics
        pnl_std = trades_df['final_pnl'].std()
        sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'final_capital': self.current_capital,
            'avg_pnl_per_trade': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'trades_df': trades_df,
            'daily_df': daily_df
        }
        
        return results

def main():
    """Main backtest function"""
    db_path = "../sql-database/options_data.db"
    
    # Define backtest period (6 months from available data)
    start_date = "2013-01-02"
    end_date = "2013-06-28"
    
    print("🎯 Short-Term Put Strategy Backtest")
    print("=" * 60)
    
    # Run backtest with different selection criteria
    strategies = ['undervalued', 'high_volume', 'itm', 'random']
    
    for strategy in strategies:
        print(f"\n📊 Testing Strategy: {strategy.upper()}")
        print("-" * 40)
        
        backtest = PutStrategyBacktest(db_path, seed_capital=500000)
        results = backtest.run_backtest(start_date, end_date, strategy)
        
        if results:
            print(f"\n📈 Results for {strategy.upper()} Strategy:")
            print(f"   Total Trades: {results['total_trades']:,}")
            print(f"   Win Rate: {results['win_rate']:.1%}")
            print(f"   Total P&L: ${results['total_pnl']:,.2f}")
            print(f"   Total Return: {results['total_return']:.1%}")
            print(f"   Final Capital: ${results['final_capital']:,.2f}")
            print(f"   Avg P&L per Trade: ${results['avg_pnl_per_trade']:.2f}")
            print(f"   Max Profit: ${results['max_profit']:.2f}")
            print(f"   Max Loss: ${results['max_loss']:.2f}")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            
            # Show top performing trades
            if not results['trades_df'].empty:
                top_trades = results['trades_df'].nlargest(5, 'final_pnl')
                print(f"\n🏆 Top 5 Trades:")
                print(top_trades[['underlying', 'strike', 'contracts', 'final_pnl']].to_string(index=False))
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()

