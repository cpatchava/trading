"""
weekly_put_strategy.py  —  v4 (with weekly summary)

Plan a weekly put-selling strategy using ranked PUTs, tracking real P/L,
applying risk filters, and summarizing weekly performance.

Outputs
-------
Writes two SQLite tables:
  1. `weekly_put_strategy`: detailed trades with columns:
     - week_start, underlying, put_stock_price, avg_bid, avg_edge,
       exp_return, contract_count, contracts_sold, allocation_per_symbol,
       premium_income, assignment_loss, net_pnl, cash_after_trade
  2. `weekly_put_summary`: weekly P/L summary with columns:
     - week_start, weekly_pnl, equity

New Filters
-----------
- **Edge threshold**: only consider symbols with `avg_edge >= edge_threshold`.
- **OTM filter**: only sell puts at least `otm_percent` out-of-the-money.

Usage
-----
```bash
python weekly_put_strategy.py \
  --db ../sql-database/options_data.db \
  --rank-table weekly_put_rank \
  --output-table weekly_put_strategy \
  --capital 500000 \
  --num-picks 10 \
  --reinvest \
  --edge-threshold 0.0 \
  --otm-percent 0.05
```
"""

import sqlite3
import pandas as pd
import argparse
import math
from typing import Tuple


def plan_weekly_puts(
    *,
    db: str,
    rank_table: str = "weekly_put_rank",
    output_table: str = "weekly_put_strategy",
    capital: float = 500_000.0,
    num_picks: int = 10,
    reinvest: bool = True,
    min_liquidity: int = 1,
    edge_threshold: float = 0.0,
    otm_percent: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and persist the weekly put-selling plan with real P/L and filters.

    Returns:
        strategy_df: detailed trade-level results
        summary_df: weekly P/L and equity summary
    """
    # 1) Load ranked puts
    with sqlite3.connect(db) as con:
        df = pd.read_sql(f"SELECT * FROM {rank_table}", con, parse_dates=["week_start"])
    if df.empty:
        raise ValueError(f"No data in ranking table {rank_table}")

    # 2) Pre-filters
    df = df[df["contract_count"] >= min_liquidity]
    df = df[df["avg_edge"] >= edge_threshold]
    df = df[df["put_stock_price"] <= df["avg_open"] * (1 - otm_percent)]

    # 3) Sort
    df.sort_values(["week_start", "exp_return", "avg_edge"], ascending=[True, False, False], inplace=True)

    results = []
    cash = capital

    # 4) Simulate weekly trades
    for week, group in df.groupby("week_start"):
        picks = group.head(num_picks)
        if picks.empty:
            # no trades this week
            results.append({
                "week_start": week,
                "underlying": None,
                "contracts_sold": 0,
                "allocation_per_symbol": 0.0,
                "premium_income": 0.0,
                "assignment_loss": 0.0,
                "net_pnl": 0.0,
                "cash_after_trade": cash,
            })
            continue

        bankroll = cash if reinvest else capital
        allocation = bankroll / num_picks

        for _, row in picks.iterrows():
            notional = row["put_stock_price"] * 100
            contracts = math.floor(allocation / notional)
            if contracts < 1 or cash < notional:
                continue

            premium = row["avg_bid"] * 100 * contracts
            assignment_loss = 0.0
            if "avg_close" in row and pd.notna(row["avg_close"]):
                loss_per_share = max(0.0, row["put_stock_price"] - row["avg_close"])
                assignment_loss = loss_per_share * 100 * contracts

            net_pnl = premium - assignment_loss
            cash += net_pnl

            results.append({
                "week_start": week,
                "underlying": row["underlying"],
                "put_stock_price": row["put_stock_price"],
                "avg_bid": row["avg_bid"],
                "avg_edge": row["avg_edge"],
                "exp_return": row["exp_return"],
                "contract_count": row["contract_count"],
                "contracts_sold": contracts,
                "allocation_per_symbol": allocation,
                "premium_income": premium,
                "assignment_loss": assignment_loss,
                "net_pnl": net_pnl,
                "cash_after_trade": cash,
            })

    strategy_df = pd.DataFrame(results)

    # 5) Build weekly summary
    summary_df = (
        strategy_df.groupby("week_start").agg(
            weekly_pnl=("net_pnl", "sum"),
            equity=("cash_after_trade", "last")
        )
        .reset_index()
    )

    # 6) Persist to SQL
    with sqlite3.connect(db) as con:
        con.execute(f"DROP TABLE IF EXISTS {output_table}")
        strategy_df.to_sql(output_table, con, if_exists="replace", index=False)
        con.execute("DROP TABLE IF EXISTS weekly_put_summary")
        summary_df.to_sql("weekly_put_summary", con, if_exists="replace", index=False)

    return strategy_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan weekly put-selling strategy with filters and real P/L and summary")
    parser.add_argument("--db",            required=True, help="SQLite DB with ranking")
    parser.add_argument("--rank-table",   required=True, help="Weekly rank table name")
    parser.add_argument("--output-table", default="weekly_put_strategy", help="Strategy table name")
    parser.add_argument("--capital",      type=float, default=500000, help="Initial capital")
    parser.add_argument("--num-picks",    type=int,   default=10,     help="Symbols to pick each week")
    parser.add_argument("--reinvest",     action="store_true",     help="Reinvest P/L into next week")
    parser.add_argument("--min-liquidity",type=int,   default=1,      help="Min contract_count to consider")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Min avg_edge to consider")
    parser.add_argument("--otm-percent",    type=float, default=0.05,help="Min % OTM for strike")
    args = parser.parse_args()

    strat_df, sum_df = plan_weekly_puts(
        db=args.db,
        rank_table=args.rank_table,
        output_table=args.output_table,
        capital=args.capital,
        num_picks=args.num_picks,
        reinvest=args.reinvest,
        min_liquidity=args.min_liquidity,
        edge_threshold=args.edge_threshold,
        otm_percent=args.otm_percent,
    )
    print(sum_df.head())
    print(strat_df.head())
