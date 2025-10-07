import sqlite3
import pandas as pd
import numpy as np


def create_bs_models(db_path: str,
                     db_path_stocks: str,
                     sp500_symbols_path: str,
                     hist_window: int = 20):
    """
    Connects to the options and stocks databases, computes Black-Scholes model inputs,
    and writes the result to a new table `bs_models` in the options database.

    Parameters:
    - db_path: Path to the SQLite options database (options_data.db)
    - db_path_stocks: Path to the SQLite stocks database (stocks_data.db)
    - sp500_symbols_path: Path to a text file listing S&P 500 symbols, one per line
    - hist_window: Lookback window (in trading days) for historical volatility calculation
    """
    # Connect to the databases
    conn_opts = sqlite3.connect(db_path)
    conn_stocks = sqlite3.connect(db_path_stocks)

    # Load tables into DataFrames with parsed dates
    opts = pd.read_sql(
        'SELECT * FROM options',
        conn_opts,
        parse_dates=['quote_date', 'expiration']
    )
    stocks = pd.read_sql(
        'SELECT * FROM stocks',
        conn_stocks,
        parse_dates=['quote_date']
    )

    # --- Compute BS model input fields ---
    # 1. Days to expiry (in years)
    opts['days_to_expiry'] = (
        opts['expiration'] - opts['quote_date']
    ).dt.days

    # 2. Mid price
    opts['mid_price'] = (opts['bid'] + opts['ask']) / 2.0

    # 3. Day of week enum (Monday=0 ... Friday=4)
    opts['day_of_week'] = opts['quote_date'].dt.dayofweek

    # 4. S&P 500 flag
    sp500 = pd.read_csv(sp500_symbols_path, header=None)[0].tolist()
    opts['is_sp500'] = opts['underlying'].isin(sp500)

    # 5. Historical volatility: compute on stock close
    stocks_sorted = stocks.sort_values(['symbol', 'quote_date'])
    stocks_sorted['log_ret'] = np.log(
        stocks_sorted['close'] / stocks_sorted.groupby('symbol')['close'].shift(1)
    )

    # Rolling std dev of log returns, annualized
    stocks_sorted['hist_vol'] = (
        stocks_sorted.groupby('symbol')['log_ret']
        .rolling(window=hist_window)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)
    )

    # Merge historical vol onto options
    hist = stocks_sorted[['symbol', 'quote_date', 'hist_vol']]
    opts = opts.merge(
        hist,
        left_on=['underlying', 'quote_date'],
        right_on=['symbol', 'quote_date'],
        how='left'
    )
    opts.drop(columns=['symbol'], inplace=True)

    # Select relevant columns for output
    bs_models = opts[
        [
            'contract', 'underlying', 'expiration', 'type', 'strike', 'style',
            'bid', 'ask', 'volume', 'open_interest', 'quote_date',
            'days_to_expiry', 'mid_price', 'day_of_week', 'hist_vol', 'is_sp500'
        ]
    ]

    # Write to SQLite (replace existing bs_models table)
    bs_models.to_sql('bs_models', conn_opts, if_exists='replace', index=False)

    # Close connections
    conn_opts.close()
    conn_stocks.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate BS model input table from options and stock data.'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='../sql-database/options_data.db',
        help='Path to the SQLite options database'
    )
    parser.add_argument(
        '--db-path-stocks',
        type=str,
        default='../sql-database/stocks_data.db',
        help='Path to the SQLite stocks database'
    )
    parser.add_argument(
        '--sp500-path',
        type=str,
        default='../sql-database/sp500_symbols.txt',
        help='Path to the text file with S&P 500 symbols'
    )
    parser.add_argument(
        '--hist-window',
        type=int,
        default=20,
        help='Lookback window (in trading days) for historical volatility'
    )
    args = parser.parse_args()

    create_bs_models(
        args.db_path,
        args.db_path_stocks,
        args.sp500_path,
        hist_window=args.hist_window
    )
