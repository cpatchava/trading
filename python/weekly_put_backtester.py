"""
weekly_put_backtester.py  —  v5.2

Back-test a weekly short-put strategy on S&P 500 tickers using model-based edge
to select top symbols each week and track detailed performance.

Strategy
--------
1. **Universe**: tickers from `sp500_symbols.txt`.
2. **Selection**: every Monday, compute **average edge** (bid - model_price) for
   each symbol with contracts expiring that Friday; choose top N symbols by
   highest average edge (even if negative).
3. **Sizing**: divide bankroll evenly among picks; sell the maximum integer
   number of put contracts (100× strike) that fit the per-symbol allocation,
   ensuring at least one contract if cash permits.
4. **Exit**: at Friday expiration, compute P/L = premium received - assignment
   loss (if stock_close < strike, loss = (strike - stock_close)*100*contracts).
5. **Capital**: start with initial capital; if `reinvest=True`, add weekly P/L
   back into bankroll; otherwise keep initial capital fixed and track cash
   separately.

Outputs
-------
- Writes two SQLite tables to `options_db`:
  1. `weekly_picks` — detailed list of each pick: week, symbol, strike,
     contracts, premium, average_edge, assigned, pnl
  2. `weekly_summary` — summary per week: week, total_pnl, equity
- Returns `(summary_df, picks_df, fig)` where `fig` is a Matplotlib Figure if
  `plot=True` else `None`.

Usage
-----
```python
from weekly_put_backtester import run_backtest

summary_df, picks_df, fig = run_backtest(
    capital=500_000,
    reinvest=True,
    months=3,
    num_picks=10,
    start_date="2013-01-07",
    options_db="../sql-database/options_data.db",
    options_table="bs_put_20d",
    sp500_file="../sql-database/sp500_symbols.txt",
    plot=True,
)
print(summary_df)
print(picks_df.head())
fig.show()
```"""

import datetime as dt
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

@dataclass
class BacktestConfig:
    capital: float
    reinvest: bool
    months: int
    num_picks: int
    start_date: Optional[str]
    options_db: str
    options_table: str
    sp500_file: str
    weekly_picks_table: str = "weekly_picks"
    weekly_summary_table: str = "weekly_summary"

    def sp_symbols(self) -> Sequence[str]:
        return [
            s.strip().upper()
            for s in Path(self.sp500_file).read_text().splitlines()
            if s.strip()
        ]


def _next_weekday(d: dt.date, weekday: int) -> dt.date:
    return d + dt.timedelta(days=(weekday - d.weekday()) % 7)


def run_backtest(
    *,
    capital: float = 500_000.0,
    reinvest: bool = True,
    months: int = 3,
    num_picks: int = 10,
    start_date: Optional[str] = None,
    options_db: str = "../sql-database/options_data.db",
    options_table: str = "bs_put_20d",
    sp500_file: str = "../sql-database/sp500_symbols.txt",
    plot: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[plt.Figure]]:
    """Execute the weekly put-selling backtest."""
    cfg = BacktestConfig(
        capital=capital,
        reinvest=reinvest,
        months=months,
        num_picks=num_picks,
        start_date=start_date,
        options_db=options_db,
        options_table=options_table,
        sp500_file=sp500_file,
    )
    syms = cfg.sp_symbols()

    # Load options data
    with sqlite3.connect(cfg.options_db) as con:
        placeholders = ",".join(["?"] * len(syms))
        opt = pd.read_sql(
            f"SELECT * FROM {cfg.options_table} WHERE underlying IN ({placeholders})",
            con,
            params=syms,
        )

    if opt.empty:
        raise ValueError(f"No data in {cfg.options_table} for S&P 500 universe.")

    # Preprocess
    opt["quote_date"] = pd.to_datetime(opt["quote_date"]).dt.date
    opt["expiration"] = pd.to_datetime(opt["expiration"]).dt.date
    opt["edge"] = opt["bid"] - opt["model_price"]
    opt["premium"] = opt["bid"] * 100
    opt["notional"] = opt["strike"] * 100

    # Determine date range
    first_date = opt["quote_date"].min()
    if start_date:
        start_d = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start_d = first_date
    start = _next_weekday(start_d, 0)
    end = start + dt.timedelta(days=months * 30)

    cash = capital
    picks_records = []
    summary_records = []

    current = start
    while current < end:
        friday = _next_weekday(current, 4)
        week_data = opt[
            (opt["quote_date"] == current) &
            (opt["expiration"] == friday)
        ]

        # Average edge per symbol
        avg_edge = (
            week_data.groupby("underlying")["edge"]
            .mean().reset_index()
        )
        top_symbols = (
            avg_edge.sort_values("edge", ascending=False)
            .head(num_picks)["underlying"]
        )

        allocation = (cash if reinvest else capital) / num_picks
        week_pnl = 0.0

        for symbol in top_symbols:
            subset = week_data[week_data["underlying"] == symbol]
            if subset.empty:
                continue
            row = subset.loc[subset["edge"].idxmax()]
            notional = row["notional"]
            contracts = math.floor(allocation / notional)
            if contracts < 1 and cash >= notional:
                contracts = 1
            if contracts < 1:
                continue

            premium = row["premium"] * contracts
            cash += premium
            assigned = row["stock_close"] < row["strike"]
            loss = (
                (row["strike"] - row["stock_close"]) * 100 * contracts
                if assigned else 0.0
            )
            pnl = premium - loss
            cash -= loss
            week_pnl += pnl

            picks_records.append({
                "week": current,
                "symbol": symbol,
                "strike": row["strike"],
                "contracts": contracts,
                "premium": premium,
                "average_edge": row["edge"],
                "assigned": assigned,
                "pnl": pnl,
            })

        summary_records.append({
            "week": current,
            "total_pnl": week_pnl,
            "equity": cash,
        })
        current += dt.timedelta(days=7)

    picks_df = pd.DataFrame(picks_records)
    summary_df = pd.DataFrame(summary_records)

    # Persist to SQL
    with sqlite3.connect(cfg.options_db) as con:
        con.execute(f"DROP TABLE IF EXISTS {cfg.weekly_picks_table}")
        if not picks_df.empty:
            picks_df.to_sql(cfg.weekly_picks_table, con, if_exists="replace", index=False)
        else:
            con.execute(
                f"CREATE TABLE {cfg.weekly_picks_table}"
                "(week DATE, symbol TEXT, strike REAL, contracts INTEGER,"
                " premium REAL, average_edge REAL, assigned BOOLEAN, pnl REAL)"
            )
        con.execute(f"DROP TABLE IF EXISTS {cfg.weekly_summary_table}")
        if not summary_df.empty:
            summary_df.to_sql(cfg.weekly_summary_table, con, if_exists="replace", index=False)
        else:
            con.execute(
                f"CREATE TABLE {cfg.weekly_summary_table}"
                "(week DATE, total_pnl REAL, equity REAL)"
            )

    # Plot
    fig = None
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].bar(
            summary_df["week"], summary_df["total_pnl"],
            color=["#4caf50" if v>=0 else "#f44336" for v in summary_df["total_pnl"]]
        )
        axes[0].set_title("Weekly P/L")
        axes[0].set_ylabel("P/L ($)")
        axes[1].plot(summary_df["week"], summary_df["equity"], marker='o')
        axes[1].set_title("Equity Curve")
        axes[1].set_ylabel("Equity ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return summary_df, picks_df, fig

# CLI
if __name__ == "__main__":
    summary_df, picks_df, fig = run_backtest(
        capital=500_000,
        reinvest=True,
        months=3,
        num_picks=10,
        start_date="2013-01-07",
        options_db="../sql-database/options_data.db",
        options_table="bs_put_20d",
        sp500_file="../sql-database/sp500_symbols.txt",
        plot=True,
    )
    print(summary_df.head())
    print(picks_df.head())
