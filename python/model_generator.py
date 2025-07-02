"""
model_generator.py

Simplified, clean version – no vectorised math, just straightforward
per‑row Black‑Scholes pricing with a fast sigma lookup.
Designed for weekly (≤ max_days_to_expiry) S&P‑500 puts.
"""

import math
import os
import sqlite3
from collections import defaultdict
from bisect import bisect_right

import pandas as pd
from scipy.stats import norm

# ────────────────────────── Black‑Scholes (European put)

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Return theoretical European put price (per share)."""
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ────────────────────────── helper: write sample S&P‑500 file if missing

def create_sp500_file(path: str = "../sql-database/sp500_symbols.txt") -> None:
    sample = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "XOM", "KO",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(sample))
    print(f"✅ Sample S&P‑500 file written → {path}")

# ────────────────────────── sigma utilities

def build_sigma_map(stk_df: pd.DataFrame, lookback: int) -> dict:
    """Return nested dict  sigma_map[sym][(yr, mo)] = (date_list, sigma_list)."""
    mmap = defaultdict(lambda: defaultdict(lambda: ([], [])))
    df = stk_df.dropna(subset=["sigma"]).copy()
    df["year"] = df["quote_date"].dt.year
    df["month"] = df["quote_date"].dt.month
    df = df.sort_values(["underlying", "quote_date"])

    for (sym, yr, mo), grp in df.groupby(["underlying", "year", "month"]):
        mmap[sym][(yr, mo)] = (
            grp["quote_date"].dt.strftime("%Y-%m-%d").tolist(),
            grp["sigma"].tolist(),
        )
    return mmap


def sigma_lookup(sym: str, qdate: pd.Timestamp, mmap: dict, fallback: bool = True):
    """Return sigma for sym using latest date ≤ qdate (month bucket fallback)."""
    yr, mo = qdate.year, qdate.month
    while True:
        dates, sigs = mmap.get(sym, {}).get((yr, mo), ([], []))
        if dates:
            idx = bisect_right(dates, qdate.strftime("%Y-%m-%d")) - 1
            if idx >= 0:
                return sigs[idx]
        if not fallback:
            return None
        # step back a month
        mo -= 1
        if mo == 0:
            yr -= 1
            mo = 12
        if yr < 1900:  # safety
            return None

# ────────────────────────── main generator

def generate_model_table(
    *,
    model_name: str = "bs_put",
    option_type: str = "put",
    r: float = 0.01,
    lookback: int = 10,
    output_table: str = "model_outputs",
    use_sp500_only: bool = True,
    max_days_to_expiry: int = 21,
):
    """Compute Black‑Scholes prices and write to <output_table>."""

    # DB connections
    opt_con = sqlite3.connect("../sql-database/options_data.db")
    stk_con = sqlite3.connect("../sql-database/stocks_data.db")

    # 1️⃣ Option filter (S&P + short expiry) --------------------------------
    params: list = [option_type, max_days_to_expiry]
    sp_filter = ""
    if use_sp500_only:
        sp_file = "../sql-database/sp500_symbols.txt"
        if not os.path.exists(sp_file):
            create_sp500_file(sp_file)
        with open(sp_file) as f:
            sp_syms = tuple(s.strip() for s in f if s.strip())
        sp_filter = f" AND underlying IN ({','.join(['?']*len(sp_syms))}) "
        params.extend(sp_syms)

    print("📥 Loading filtered options …")
    query = f"""
        SELECT * FROM options
        WHERE type = ?
          AND julianday(expiration) - julianday(quote_date) <= ?
          {sp_filter}
    """
    opt_df = pd.read_sql(query, opt_con, params=params)
    if opt_df.empty:
        print("⚠️  No option rows matched filter; aborting.")
        return

    # ensure date columns are datetime for safe merging
    opt_df["quote_date"] = pd.to_datetime(opt_df["quote_date"])
    opt_df["expiration"] = pd.to_datetime(opt_df["expiration"])

    opt_df["bid"] /= 100.0
    opt_df["ask"] /= 100.0

    # 2️⃣ Stock closes & returns -------------------------------------------
    stocks = pd.read_sql("SELECT symbol, quote_date, close FROM stocks", stk_con)
    stocks = stocks.rename(columns={"symbol": "underlying", "close": "stock_close"})
    stocks["quote_date"] = pd.to_datetime(stocks["quote_date"])
    stocks = stocks.sort_values(["underlying", "quote_date"])
    stocks["ret"] = stocks.groupby("underlying")["stock_close"].pct_change()
    stocks["sigma"] = (
        stocks.groupby("underlying")["ret"]
        .rolling(window=lookback, min_periods=lookback)
        .std()
        .reset_index(level=0, drop=True)
        * math.sqrt(252)
    )

    # 3️⃣ Merge & sigma lookup map ----------------------------------------
    merged = opt_df.merge(stocks, on=["underlying", "quote_date"], how="left")
    sigma_map = build_sigma_map(stocks, lookback)

    # 4️⃣ Loop & compute model prices -------------------------------------
    model_prices = []
    for _, row in merged.iterrows():
        sigma = sigma_lookup(
            row["underlying"], pd.to_datetime(row["quote_date"]), sigma_map
        )
        if sigma is None:
            model_prices.append(None)
            continue
        S = row["stock_close"];  K = row["strike"]
        T = (row["expiration"] - row["quote_date"]).days / 365.0
        model_prices.append(black_scholes_put(S, K, T, r, sigma))
        print("Model Price %s, Stock Close %s, Days to expiration %s, Strike Price %s", (model_prices[-1], S, T, K))
    merged["model_price"] = model_prices

    # 5️⃣ Write out --------------------------------------------------------
    out_cols = [
        "contract", "underlying", "quote_date", "expiration", "strike",
        "type", "bid", "ask", "stock_close", "model_price",
    ]
    with sqlite3.connect("../sql-database/options_data.db") as conn:
        merged[out_cols].to_sql(output_table, conn, if_exists="replace", index=False)

    print(f"✅ Wrote {merged[out_cols].shape[0]} rows → table '{output_table}'")

