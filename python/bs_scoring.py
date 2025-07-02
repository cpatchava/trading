import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm


def generate_bs_scores(options_db_path: str,
                       stocks_db_path: str,
                       risk_free_rate: float = 0.01,
                       dividend_yield: float = 0.0):
    """
    Reads the `bs_models` table from the options database, joins with stock prices,
    computes Black-Scholes theoretical prices and pricing differences, and writes
    the results to a new `bs_scores` table in the options database.

    Parameters:
    - options_db_path: Path to the SQLite options database containing `bs_models`
    - stocks_db_path: Path to the SQLite stocks database containing `stocks`
    - risk_free_rate: Annual risk-free rate (e.g., 0.01 for 1%)
    - dividend_yield: Annual dividend yield (e.g., 0.02 for 2%)
    """
    # Connect to databases
    conn_opts = sqlite3.connect(options_db_path)
    conn_stocks = sqlite3.connect(stocks_db_path)

    # Load tables
    bs_df = pd.read_sql('SELECT * FROM bs_models', conn_opts,
                        parse_dates=['quote_date', 'expiration'])
    stocks_df = pd.read_sql('SELECT symbol, close, quote_date FROM stocks',
                            conn_stocks, parse_dates=['quote_date'])

    # Merge to get underlying price
    merged = bs_df.merge(
        stocks_df.rename(columns={'symbol': 'underlying'}),
        on=['underlying', 'quote_date'], how='left'
    )

    # Prepare variables
    S = merged['close'].values
    K = merged['strike'].values
    # days_to_expiry is in days; convert to years
    T = merged['days_to_expiry'].values / 365.0
    sigma = merged['hist_vol'].values
    r = risk_free_rate
    q = dividend_yield

    # Black-Scholes d1, d2
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # Compute theoretical prices
    is_call = merged['type'] == 'call'
    bs_price = np.where(
        is_call,
        S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
        K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    )

    # Compute mid-price difference (market - model)
    price_diff = merged['mid_price'].values - bs_price

    # Assign back to DataFrame
    merged['bs_price'] = bs_price
    merged['price_diff'] = price_diff

    # Select columns and write to new table
    out_cols = [
        'contract', 'underlying', 'quote_date', 'expiration', 'type', 'strike',
        'mid_price', 'bs_price', 'price_diff'
    ]
    merged[out_cols].to_sql('bs_scores', conn_opts, if_exists='replace', index=False)

    # Close connections
    conn_opts.close()
    conn_stocks.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Black-Scholes model scores for options.'
    )
    parser.add_argument(
        '--options-db',
        type=str,
        required=True,
        help='Path to the options SQLite database (with bs_models)'
    )
    parser.add_argument(
        '--stocks-db',
        type=str,
        required=True,
        help='Path to the stocks SQLite database'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.01,
        help='Annual risk-free rate (e.g., 0.01 for 1%)'
    )
    parser.add_argument(
        '--dividend-yield',
        type=float,
        default=0.0,
        help='Annual dividend yield (e.g., 0.02 for 2%)'
    )
    args = parser.parse_args()

    generate_bs_scores(
        args.options_db,
        args.stocks_db,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield
    )
