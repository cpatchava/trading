#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import math
from math import erf
import argparse
import time

# volatility buckets → column names
WINDOW_THRESHOLDS = [
    (7,  'volatility_1_week'),
    (14, 'volatility_2_weeks'),
    (30, 'volatility_1_month'),
    (float('inf'), 'volatility_3_months'),
]

# Vectorized Normal CDF
_erf_vec = np.vectorize(erf)
def norm_cdf(x):
    return 0.5 * (1.0 + _erf_vec(x / math.sqrt(2.0)))

def black_scholes_vectorized(S, K, T, r, sigma, is_put):
    price = np.zeros_like(S, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
        d1 = np.zeros_like(S); d2 = np.zeros_like(S)
        d1[valid] = (np.log(S[valid]/K[valid]) + (r + 0.5*sigma[valid]**2)*T[valid]) \
                    / (sigma[valid]*np.sqrt(T[valid]))
        d2[valid] = d1[valid] - sigma[valid]*np.sqrt(T[valid])

        c1  = norm_cdf(d1);   c2  = norm_cdf(d2)
        cn1 = norm_cdf(-d1);  cn2 = norm_cdf(-d2)
        call = S*c1 - K*np.exp(-r*T)*c2
        put  = K*np.exp(-r*T)*cn2 - S*cn1

        price[valid] = np.where(is_put[valid], put[valid], call[valid])
    return price

def main(options_db: str, risk_free_rate: float, chunk_size: int):
    conn = sqlite3.connect(options_db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")

    # Ensure the pricing column exists
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE enriched_options ADD COLUMN black_scholes_model_price REAL;")
        print("➕ Added column black_scholes_model_price")
    except sqlite3.OperationalError:
        print("🟢 Column black_scholes_model_price already exists")
    conn.commit()

    # Count total & pending
    total = cur.execute("SELECT COUNT(*) FROM enriched_options").fetchone()[0]
    pending = cur.execute(
        "SELECT COUNT(*) FROM enriched_options WHERE black_scholes_model_price IS NULL"
    ).fetchone()[0]
    print(f"📊 Total rows: {total}, Pending to price: {pending}")
    if pending == 0:
        print("✅ Nothing to do.")
        return

    # Prepare update statement
    update_sql = """
      UPDATE enriched_options
      SET black_scholes_model_price = ?
      WHERE rowid = ?
    """

    # Read & process in chunks
    reader = pd.read_sql_query(
        """
        SELECT
          rowid,
          strike,
          type,
          num_days,
          stock_price_today,
          volatility_1_week,
          volatility_2_weeks,
          volatility_1_month,
          volatility_3_months
        FROM enriched_options
        WHERE black_scholes_model_price IS NULL
        """,
        conn,
        chunksize=chunk_size
    )

    start_time = time.time()
    chunk_idx = 0

    for df in reader:
        chunk_idx += 1
        n = len(df)
        print(f"\n▶️ Chunk #{chunk_idx}: {n} rows")

        # Determine sigma for each row
        conds   = [df['num_days'] <= t for t,_ in WINDOW_THRESHOLDS[:-1]]
        choices = [df[col]               for _,col in WINDOW_THRESHOLDS[:-1]]
        default = df[WINDOW_THRESHOLDS[-1][1]]
        sigma   = np.select(conds, choices, default=default).astype(float)

        # Prepare arrays
        T       = (df['num_days'].values / 365.25).astype(float)
        S       = df['stock_price_today'].astype(float).values
        K       = df['strike'].astype(float).values
        is_put  = (df['type'].str.lower() == 'put').values
        rowids  = df['rowid'].astype(int).values

        # Compute prices
        t0 = time.time()
        prices = black_scholes_vectorized(S, K, T, risk_free_rate, sigma, is_put)
        t_compute = time.time() - t0

        # Batch update
        updates = list(zip(prices.tolist(), rowids.tolist()))
        t1 = time.time()
        conn.executemany(update_sql, updates)
        conn.commit()
        t_write = time.time() - t1

        print(f"   ⚙️  Compute: {t_compute:.2f}s, Write: {t_write:.2f}s")

    total_time = time.time() - start_time
    print(f"\n✅ Completed pricing {pending} rows in {total_time:.2f}s")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunked, vectorized Black–Scholes pricing on enriched_options"
    )
    parser.add_argument(
        "--options-db", default="../sql-database/options_data.db",
        help="Path to options SQLite DB"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.01,
        help="Continuous annual risk-free rate (e.g. 0.01)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Rows to process per batch"
    )
    args = parser.parse_args()

    main(
        options_db=args.options_db,
        risk_free_rate=args.risk_free_rate,
        chunk_size=args.chunk_size
    )
