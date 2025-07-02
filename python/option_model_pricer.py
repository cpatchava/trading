"""
option_model_pricer.py  —  v2

Compute Black‑Scholes fair‑value for option contracts using historical σ from
`sigma_table`. If the options table does **not** already contain a
`stock_close` column, we automatically pull the matching close price from a
`stocks` table in `stocks_db` (symbol+quote_date) before pricing.

Only S&P 500 tickers (listed in `../sql-database/sp500_symbols.txt`) are
processed to keep runtime reasonable.
"""

from __future__ import annotations

import sqlite3
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

# ────────────────────────────────────────────────────────────────────────────
# Black‑Scholes helpers
# ────────────────────────────────────────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma):
    sigma_sqrt_T = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def _bs_put(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_call(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(-d2)


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def _load_sp500(path: str) -> Sequence[str]:
    with open(path) as f:
        syms = [s.strip().upper() for s in f if s.strip()]
    if not syms:
        raise ValueError("S&P 500 ticker file is empty.")
    return syms


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def price_options(
    *,
    sigma_table: str,
    option_type: Literal["put", "call"] = "put",
    r: float = 0.01,
    max_days_to_expiry: int = 5,
    sp500_file: str = "../sql-database/sp500_symbols.txt",
    options_db: str = "../sql-database/options_data.db",
    stocks_db: str = "../sql-database/stocks_data.db",
    options_table: str = "options",
    stocks_table: str = "stocks",
    output_table: Optional[str] = None,
    min_days_to_expiry: int = 4,
) -> str:
    """Black‑Scholes price S&P 500 option contracts.

    * If `stock_close` column is missing in *options_table*, the function joins
      against *stocks_table* (`symbol`, `quote_date`, `close`) to fetch it.
    * Results are written to *output_table* (replaced on each run).
    """

    output_table = output_table or f"bs_{option_type}"
    sp_syms = _load_sp500(sp500_file)

    # 1️⃣  Load σ values --------------------------------------------------
    with sqlite3.connect(stocks_db) as con:
        sigmas = pd.read_sql(
            f"SELECT symbol AS underlying, quote_date, sigma FROM {sigma_table}", con
        )
    sigmas["quote_date"] = pd.to_datetime(sigmas["quote_date"])

    # 2️⃣  Load option quotes --------------------------------------------
    ph = ",".join(["?"] * len(sp_syms))
    params = [option_type, max_days_to_expiry, min_days_to_expiry] + sp_syms
    query = f"""
        SELECT * FROM {options_table}
        WHERE type = ?
          AND julianday(expiration) - julianday(quote_date) <= ?
          AND julianday(expiration) - julianday(quote_date) > ? 
          AND underlying IN ({ph})
    """
    with sqlite3.connect(options_db) as con:
        opt_df = pd.read_sql(query, con, params=params)

    if opt_df.empty:
        raise ValueError("No option rows matched filters.")

    opt_df["quote_date"] = pd.to_datetime(opt_df["quote_date"])
    opt_df["expiration"] = pd.to_datetime(opt_df["expiration"])

    # Convert cents→dollars if clearly needed
    if opt_df[["bid", "ask"]].max().max() > 50:
        opt_df[["bid", "ask"]] /= 100.0

    # 3️⃣  Ensure stock_close present ------------------------------------
    if "stock_close" not in opt_df.columns:
        with sqlite3.connect(stocks_db) as con:
            closes = pd.read_sql(
                f"SELECT symbol AS underlying, quote_date, close AS stock_close FROM {stocks_table}",
                con,
            )
        closes["quote_date"] = pd.to_datetime(closes["quote_date"])
        opt_df = opt_df.merge(closes, on=["underlying", "quote_date"], how="left")

    if opt_df["stock_close"].isna().any():
        raise ValueError("Missing stock_close for some option rows after merge.")

    # 4️⃣  Merge σ --------------------------------------------------------
    merged = opt_df.merge(sigmas, on=["underlying", "quote_date"], how="left")

    # 5️⃣  Calculate model price ----------------------------------------
    mask = merged["sigma"].notna()
    if mask.any():
        T = (merged.loc[mask, "expiration"] - merged.loc[mask, "quote_date"]).dt.days / 365
        price_fn = _bs_put if option_type == "put" else _bs_call
        merged.loc[mask, "model_price"] = price_fn(
            merged.loc[mask, "stock_close"],
            merged.loc[mask, "strike"],
            T,
            r,
            merged.loc[mask, "sigma"],
        )
    merged["edge"] = merged["bid"] - merged["model_price"]

    # 6️⃣  Persist --------------------------------------------------------
    cols = [
        "contract", "underlying", "quote_date", "expiration", "strike",
        "type", "bid", "ask", "stock_close", "sigma", "model_price", "edge",
    ]
    print("Finished calculating option prices.")
    with sqlite3.connect(options_db) as con:
        merged[cols].to_sql(
            output_table,
            con,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=300) # prevents “too many SQL variables” in SQLite

    return output_table


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Price S&P 500 options with Black‑Scholes.")
    p.add_argument("--sigma-table", required=True)
    p.add_argument("--option-type", choices=["put", "call"], default="put")
    p.add_argument("--max-dte", type=int, default=21)
    p.add_argument("--rf", type=float, default=0.01)
    p.add_argument("--output-table", default=None)
    p.add_argument("--options-db", default="../sql-database/options_data.db")
    p.add_argument("--stocks-db", default="../sql-database/stocks_data.db")
    p.add_argument("--options-table", default="options")
    p.add_argument("--stocks-table", default="stocks")
    p.add_argument("--sp500-file", default="../sql-database/sp500_symbols.txt")

    args = p.parse_args()

    tbl = price_options(
        sigma_table=args.sigma_table,
        option_type=args.option_type,
        r=args.rf,
        max_days_to_expiry=args.max_dte,
        sp500_file=args.sp500_file,
        options_db=args.options_db,
        stocks_db=args.stocks_db,
        options_table=args.options_table,
        stocks_table=args.stocks_table,
        output_table=args.output_table,
        min_days_to_expiry=4,
    )
    print("✅ Wrote model prices to", tbl)
