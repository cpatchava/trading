"""
sigma_generator.py

Utility to compute rolling historical volatility (annualised) for each symbol
in a daily prices table and persist it as an SQL table.

Assumes an SQLite database that already contains a daily‑prices table with
columns:
    symbol (TEXT)
    quote_date (DATE as ISO‑formatted string)
    close (REAL)

Example usage
-------------
from sigma_generator import generate_sigma_table
generate_sigma_table(lookback=20)
' sigma_lbk20'

This will add / refresh the table `sigma_lbk20` in the same database, filled
with three columns (symbol, quote_date, sigma).  Each run is idempotent: rows
are inserted with `ON CONFLICT REPLACE`, so re‑running updates existing values
and appends any new dates that have appeared.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Optional

import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def generate_sigma_table(
    *,
    lookback: int = 10,
    stocks_db: str = "../sql-database/stocks_data.db",
    stocks_table: str = "stocks",
    output_table: Optional[str] = None,
    trading_days: int = 252,
    chunksize: int = 300,  # 300 rows × 3 cols = 900 bind vars < SQLite 999 limit
) -> str:
    """Compute rolling annualised volatility and persist to an SQLite table.

    Parameters
    ----------
    lookback : int
        Rolling window length in *trading* days used for the σ calculation.
    stocks_db : str
        Path to the SQLite database that houses the *stocks_table*.
    stocks_table : str
        Name of the source daily‑prices table. Must contain columns
        ``symbol``, ``quote_date`` and ``close``.
    output_table : str | None
        Name of the table to (create and) write σ values into.  Defaults to
        ``f"sigma_lbk{lookback}"``.
    trading_days : int
        Number of trading days assumed in a year when annualising σ.
    chunksize : int
        Number of rows per batched INSERT when writing to SQLite.  Keep
        ``chunksize × number_of_columns < 999`` to stay under SQLite's bind‑
        variable limit.

    Returns
    -------
    str
        The name of the table that now contains the σ values.
    """

    output_table = output_table or f"sigma_lbk{lookback}"

    # ── Load source price data ────────────────────────────────────────────
    with sqlite3.connect(stocks_db) as con:
        df = pd.read_sql(
            f"SELECT DISTINCT symbol, quote_date, close FROM {stocks_table}", con
        )

    if df.empty:
        raise ValueError(
            f"Source table '{stocks_table}' in '{stocks_db}' is empty or missing."
        )

    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df = df.sort_values(["symbol", "quote_date"]).reset_index(drop=True)

    # ── Calculate daily returns and rolling σ per symbol ──────────────────
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    df["sigma"] = (
        df.groupby("symbol")["ret"]
        .rolling(window=lookback, min_periods=lookback)
        .std()
        .reset_index(level=0, drop=True)
        * math.sqrt(trading_days)
    ).round(6)

    out = df.dropna(subset=["sigma"])[["symbol", "quote_date", "sigma"]]

    # ── Persist idempotently ──────────────────────────────────────────────
    with sqlite3.connect(stocks_db) as con:
        # Small PRAGMA tweaks for faster bulk inserts
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {output_table} (
                symbol      TEXT NOT NULL,
                quote_date  DATE NOT NULL,
                sigma       REAL NOT NULL,
                UNIQUE(symbol, quote_date) ON CONFLICT REPLACE
            );
            """
        )

        out.to_sql(
            output_table,
            con,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )

    return output_table


# ────────────────────────────────────────────────────────────────────────────
# Simple CLI entry‑point for ad‑hoc runs
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Compute rolling σ for every symbol and persist to SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
                python sigma_generator.py --lookback 20 --db ../sql-database/stocks_data.db
            """,
        ),
    )
    parser.add_argument("--lookback", type=int, default=10, help="Rolling window length.")
    parser.add_argument("--db", default="../sql-database/stocks_data.db", help="Path to SQLite DB.")
    parser.add_argument("--stocks-table", default="stocks", help="Source prices table name.")
    parser.add_argument(
        "--output-table", default=None, help="Destination table name (defaults to sigma_lbk<lookback>)."
    )

    args = parser.parse_args()

    tbl_name = generate_sigma_table(
        lookback=args.lookback,
        stocks_db=args.db,
        stocks_table=args.stocks_table,
        output_table=args.output_table,
    )
    print(f"✅ Sigma table '{tbl_name}' updated or created in {args.db}.")
