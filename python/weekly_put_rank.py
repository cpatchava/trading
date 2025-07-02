"""
weekly_put_rank.py  —  v4.2

Generate a weekly ranking of S&P 500 put contracts based on average bid, edge,
stock prices at opening/expiration, and strike price (`put_stock_price`).

Outputs
-------
Writes a SQLite table `weekly_put_rank` with columns:
  * `week_start`       — Monday date
  * `underlying`       — ticker symbol
  * `avg_bid`          — mean bid price of all same-week expiries
  * `avg_edge`         — mean edge (bid − model_price)
  * `avg_open`         — average stock closing price at quote_date
  * `avg_close`        — average stock closing price at expiration date
  * `contract_count`   — number of contracts considered
  * `exp_return`       — expected return = avg_bid / avg_open
  * `put_stock_price`  — average strike price for the sold puts

Usage
-----
```bash
python weekly_put_rank.py \
  --options-db ../sql-database/options_data.db \
  --options-table bs_put_20d \
  --stocks-db  ../sql-database/stocks_data.db \
  --output-table weekly_put_rank
```

Then query:
```sql
SELECT *
FROM weekly_put_rank
ORDER BY week_start, exp_return DESC;
```
"""

import argparse
import sqlite3
import pandas as pd


def _monday_of(date) -> pd.Timestamp:
    """Convert input to Timestamp and return the Monday on or before it."""
    date_ts = pd.to_datetime(date)
    return (date_ts - pd.Timedelta(days=date_ts.weekday())).floor('D')


def rank_weekly_puts(
    *,
    options_db: str,
    options_table: str,
    stocks_db: str,
    output_table: str = "weekly_put_rank",
) -> pd.DataFrame:
    """Compute and persist weekly PUT rankings."""
    # 1) Load option data
    with sqlite3.connect(options_db) as con:
        df = pd.read_sql(
            f"SELECT underlying, quote_date, expiration, bid, model_price, strike, stock_close "
            f"FROM {options_table}",
            con,
            parse_dates=["quote_date", "expiration"],
        )

    # 2) Load expiration stock closes
    with sqlite3.connect(stocks_db) as con:
        stocks = pd.read_sql(
            "SELECT symbol AS underlying, quote_date, close FROM stocks",
            con,
            parse_dates=["quote_date"],
        )
    stocks["quote_date"] = stocks["quote_date"].dt.date
    stocks.rename(columns={"close": "exp_close"}, inplace=True)

    # 3) Prepare main DataFrame
    df["quote_date"] = df["quote_date"].dt.date
    df["expiration"] = df["expiration"].dt.date
    df["dte"] = (pd.to_datetime(df["expiration"]) - pd.to_datetime(df["quote_date"]))
    df["dte"] = df["dte"].dt.days
    df = df[(df["dte"] > 0) & (df["dte"] <= 7)]
    df["week_start"] = df["quote_date"].apply(_monday_of)

    # Merge in expiration close, keep quote_date close as open_close
    df = df.merge(
        stocks,
        left_on=["underlying", "expiration"],
        right_on=["underlying", "quote_date"],
        how="left",
    ).rename(columns={"stock_close": "open_close"})

    # 4) Aggregate per week & symbol
    summary = (
        df.groupby(["week_start", "underlying"]).agg(
            avg_bid=("bid", "mean"),
            avg_edge=("model_price", lambda x: (df.loc[x.index, "bid"] - x).mean()),
            avg_open=("open_close", "mean"),
            avg_close=("exp_close", "mean"),
            contract_count=("bid", "count"),
            put_stock_price=("strike", "mean"),
        )
        .reset_index()
    )
    summary["exp_return"] = summary["avg_bid"] / summary["avg_open"]

    # 5) Persist to SQLite
    with sqlite3.connect(options_db) as con:
        con.execute(f"DROP TABLE IF EXISTS {output_table}")
        summary.to_sql(
            output_table,
            con,
            if_exists="replace",
            index=False,
            method="multi",
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank weekly puts with extra metrics")
    parser.add_argument("--options-db", required=True, help="SQLite DB with options")
    parser.add_argument("--options-table", required=True, help="Options table name")
    parser.add_argument("--stocks-db", default="../sql-database/stocks_data.db", help="SQLite DB with stocks (default ../sql-database/stocks_data.db)")
    parser.add_argument("--output-table", default="weekly_put_rank", help="Destination table name")
    args = parser.parse_args()

    df = rank_weekly_puts(
        options_db=args.options_db,
        options_table=args.options_table,
        stocks_db=args.stocks_db,
        output_table=args.output_table,
    )
    print(df.head())
