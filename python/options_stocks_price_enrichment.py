#!/usr/bin/env python3
import sqlite3
import pandas as pd
import argparse

def enrich_option_prices(
    options_db: str,
    stocks_db:  str,
    chunk_size: int = 50000
):
    # 1) Build price dict
    with sqlite3.connect(stocks_db) as cs:
        df = pd.read_sql_query(
            "SELECT symbol, quote_date, close FROM stocks_enriched",
            cs, parse_dates=["quote_date"]
        )
    df["d_str"] = df["quote_date"].dt.strftime("%Y-%m-%d")
    price_dict = {
        (row.symbol, row.d_str): row.close
        for row in df.itertuples()
    }
    print(f"🔑 Loaded {len(price_dict)} prices into memory")

    # 2) Open options DB & add columns if needed
    co = sqlite3.connect(options_db)
    cur = co.cursor()
    for col in ("stock_price_today REAL", "stock_price_contract_close REAL"):
        try:
            cur.execute(f"ALTER TABLE enriched_options ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    co.commit()

    # 3) Find max rowid
    max_rid = cur.execute(
        "SELECT MAX(rowid) FROM enriched_options"
    ).fetchone()[0] or 0
    print(f"🔢 Table has {max_rid} rows; processing in chunks of {chunk_size}")

    # 4) Loop by rowid chunks
    last = 0
    while last < max_rid:
        lo, hi = last + 1, last + chunk_size
        df_opt = pd.read_sql_query(
            """
            SELECT rowid, underlying, quote_date, expiration
            FROM enriched_options
            WHERE rowid BETWEEN ? AND ?
            """,
            co,
            params=(lo, hi),
            parse_dates=["quote_date", "expiration"]
        )
        if df_opt.empty:
            last = hi
            continue

        # prepare lookup keys
        df_opt["qd"] = df_opt["quote_date"].dt.strftime("%Y-%m-%d")
        df_opt["ed"] = df_opt["expiration"].dt.strftime("%Y-%m-%d")

        # vectorized lookup
        df_opt["stock_price_today"] = [
            price_dict.get((sym, d), None)
            for sym, d in zip(df_opt["underlying"], df_opt["qd"])
        ]
        df_opt["stock_price_contract_close"] = [
            price_dict.get((sym, d), None)
            for sym, d in zip(df_opt["underlying"], df_opt["ed"])
        ]

        # batch update
        updates = [
            (row.stock_price_today,
             row.stock_price_contract_close,
             int(row.rowid))
            for row in df_opt.itertuples()
        ]
        co.executemany(
            "UPDATE enriched_options "
            "SET stock_price_today = ?, stock_price_contract_close = ? "
            "WHERE rowid = ?",
            updates
        )
        co.commit()

        print(f"↳ Updated rowid {lo}–{hi} ({len(updates)} rows)")
        last = hi

    co.close()
    print("✅ Finished price enrichment.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Add stock_price_today & stock_price_contract_close to enriched_options"
    )
    p.add_argument(
        "--options-db",
        default="../sql-database/options_data.db",
        help="Path to options SQLite DB"
    )
    p.add_argument(
        "--stocks-db",
        default="../sql-database/stocks_data.db",
        help="Path to stocks_enriched SQLite DB"
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of rows to process per batch"
    )
    args = p.parse_args()
    enrich_option_prices(
        options_db=args.options_db,
        stocks_db=args.stocks_db,
        chunk_size=args.chunk_size
    )
