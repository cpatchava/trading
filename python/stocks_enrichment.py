import os
import sqlite3
import pandas as pd
import numpy as np

# Paths
db_path = os.path.join('..', 'sql-database', 'stocks_data.db')
sp500_file = os.path.join('..', 'sql-database', 'sp500_symbols.txt')

# Time windows for volatility (in trading days)
WINDOWS = {
    'volatility_1_week': 5,
    'volatility_2_weeks': 10,
    'volatility_1_month': 21,
    'volatility_3_months': 63
}


def load_sp500_symbols(path: str) -> set:
    """
    Load S&P 500 symbols from a text file, one per line.
    """
    with open(path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def enrich_stocks_table(db_path: str, sp500_symbols: set):
    """
    Create or replace the `stocks_enriched` table that includes original data,
    a boolean `is_sp500` flag, and rolling volatility columns.
    """
    conn = sqlite3.connect(db_path)

    # Load raw stocks data
    df = pd.read_sql_query('SELECT * FROM stocks', conn)
    df['quote_date'] = pd.to_datetime(df['quote_date'], errors='coerce')

    # Filter out invalid dates and close prices
    df = df.dropna(subset=['quote_date', 'close']).copy()
    df = df[df['close'] > 0]

    # Flag S&P 500 membership
    df['is_sp500'] = df['symbol'].isin(sp500_symbols)

    # Sort by symbol and date for rolling calculations
    df.sort_values(['symbol', 'quote_date'], inplace=True)

    # Calculate daily log returns (preserves index)
    df['log_return'] = df.groupby('symbol')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df['log_return'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate rolling volatility and annualize
    for col, window in WINDOWS.items():
        df[col] = (
            df.groupby('symbol')['log_return']
              .transform(lambda x: x.rolling(window, min_periods=1).std(ddof=0) * np.sqrt(252))
        )

    # Drop the intermediate log_return column
    df.drop(columns=['log_return'], inplace=True)

    # Write enriched data to a new table
    conn.execute('DROP TABLE IF EXISTS stocks_enriched')
    df.to_sql('stocks_enriched', conn, index=False)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # Ensure DB directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Load S&P 500 symbols
    syms = load_sp500_symbols(sp500_file)
    print(f"Loaded {len(syms)} S&P 500 symbols.")

    # Enrich stocks data and write to DB
    enrich_stocks_table(db_path, syms)
    print("✅ 'stocks_enriched' table created with SP500 flag and volatility metrics.")
