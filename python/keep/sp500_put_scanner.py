"""
sp500_put_scanner.py
────────────────────────────────────────────────────────────────
Screen S&P-500 weekly put options, price with Black-Scholes,
back-test Monday-to-Friday performance, and draw weekly PnL.
"""

import os, math, sqlite3, pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  CONFIG  – adjust to your actual database folder
# ──────────────────────────────────────────────────────────────
DB_DIR     = "../sql-database"
OPTIONS_DB = os.path.join(DB_DIR, "options_data.db")
STOCKS_DB  = os.path.join(DB_DIR, "stocks_data.db")

# ──────────────────────────────────────────────────────────────
#  S&P-500 ticker list  (scraped once)
# ──────────────────────────────────────────────────────────────
def _get_sp500_set() -> set[str]:
    tbl = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return set(tbl["Symbol"].str.replace(".", "-", regex=False))

SP500_SET = _get_sp500_set()

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _connect():
    return sqlite3.connect(OPTIONS_DB), sqlite3.connect(STOCKS_DB)


def _bs_put(S, K, T, r, sigma):
    """European put (Black-Scholes)."""
    if min(S, K, T, sigma) <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _price_cols_to_dollars(df: pd.DataFrame):
    """Convert option price columns that are stored in cents → dollars."""
    for col in ("bid", "ask", "last", "mid"):
        if col in df.columns:
            df[col] = df[col] / 100.0
    return df




# add this to sp500_put_scanner.py  (name it however you like)
def weekly_put_screen_aggregate(first_monday="2013-01-07", *,
                                weeks=4, top_n=20,
                                r=0.05, lookback=10) -> pd.DataFrame:
    """
    Aggregates ALL Friday-quoted puts per ticker (expiring next Friday),
    computes the average edge = bid − Black-Scholes price.
    Positive edge ⇒ options rich ⇒ attractive to SELL.
    Returns Top-N tickers each week (one row per ticker).
    """
    opt_con, stk_con = _connect_dbs()
    start_mon = datetime.strptime(first_monday, "%Y-%m-%d")
    prior_fri = start_mon - timedelta(days=3)

    rows = []
    for w in range(weeks):
        q_fri = prior_fri + timedelta(weeks=w)          # quote date
        x_fri = q_fri   + timedelta(days=7)             # expiry
        qstr, xstr = q_fri.strftime("%Y-%m-%d"), x_fri.strftime("%Y-%m-%d")

        puts = pd.read_sql(f"""
            SELECT * FROM options
            WHERE  quote_date = '{qstr}'
              AND  type       = 'put'
              AND  expiration = '{xstr}'
        """, opt_con)
        stocks = pd.read_sql(f"""
            SELECT symbol, close AS stock_close
            FROM   stocks
            WHERE  quote_date = '{qstr}'
        """, stk_con)
        if puts.empty or stocks.empty:
            continue

        # cents → dollars
        for col in ("bid", "ask", "last", "mid"):
            if col in puts.columns:
                puts[col] = puts[col] / 100.0

        merged = (puts.merge(stocks, left_on="underlying", right_on="symbol")
                       .query("underlying in @SP500_SET")
                       .drop_duplicates(subset=['underlying','strike','bid']))
        if merged.empty:
            continue

        # volatility lookup
        sig = {}
        for sym in merged["underlying"].unique():
            hist = pd.read_sql(f"""
                SELECT close FROM stocks
                WHERE  symbol='{sym}' AND quote_date<='{qstr}'
                ORDER BY quote_date DESC LIMIT {lookback}
            """, stk_con)
            if len(hist) > 1:
                ret = hist["close"].pct_change().dropna()
                sig[sym] = ret.std() * math.sqrt(252)

        T = 7 / 365.0
        merged["sigma"] = merged["underlying"].map(sig)
        merged = merged[merged["sigma"].notna() & (merged["bid"] > 0)]

        # 🔑 FIX — pass the risk-free *r* (float) not the row object
        merged["model"] = merged.apply(
            lambda row: _bs_put(row["stock_close"],
                                row["strike"],
                                T,
                                r,                 # correct risk-free rate
                                row["sigma"]),
            axis=1)

        merged["edge_sell"] = merged["bid"] - merged["model"]

        # aggregate per ticker
        agg = (merged.groupby("underlying")
                     .agg(avg_edge  = ("edge_sell", "mean"),
                          contracts = ("strike",    "count"),
                          avg_bid   = ("bid",       "mean"),
                          avg_model = ("model",     "mean"))
                     .reset_index())
        agg["week"] = (q_fri + timedelta(days=3)).strftime("%Y-%m-%d")  # Monday label

        top = (agg[agg["avg_edge"] > 0]                 # overpriced = good to sell
               .sort_values("avg_edge", ascending=False)
               .head(top_n))
        rows.append(top)

    opt_con.close(); stk_con.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def backtest_sell_puts(picks: pd.DataFrame):
    opt_con, stk_con = _connect_dbs()
    results = []

    for _, row in picks.iterrows():
        friday_close = pd.read_sql(f"""
            SELECT close FROM stocks
            WHERE symbol='{row['ticker']}' AND quote_date='{row['expiration']}'
            LIMIT 1
        """, stk_con)
        if friday_close.empty:
            continue

        S_fri = friday_close.iat[0, 0]
        payoff = max(0, row["strike"] - S_fri) * 100      # cost to seller
        premium = row["market_bid"] * 100                 # received
        pnl = premium - payoff

        results.append({
            **row.to_dict(),
            "S_fri"  : S_fri,
            "premium": premium,
            "payoff" : payoff,
            "PnL"    : pnl,
            "assigned": payoff > 0
        })

    opt_con.close(); stk_con.close()
    trades = pd.DataFrame(results)

    weekly = (trades.groupby("week")
              .agg(total_pnl   = ("PnL","sum"),
                   assignments = ("assigned","sum"),
                   tickers     = ("ticker","count"))
              .reset_index())
    return trades, weekly

# ──────────────────────────────────────────────────────────────
#  1. Weekly screener
# ──────────────────────────────────────────────────────────────
def weekly_put_screen(start_date="2013-01-07", *,
                      weeks=4, top_n=20,
                      r=0.05, lookback=10) -> pd.DataFrame:
    """
    Screen S&P-500 weekly puts: Monday quote, Friday expiry.
    Returns the top-N most undervalued (model > bid) per week.
    """
    opt_con, stk_con = _connect_dbs()
    monday0  = datetime.strptime(start_date, "%Y-%m-%d")
    all_rows = []

    for w in range(weeks):
        wk_start = monday0 + timedelta(weeks=w)           # Monday
        fri_date = wk_start + timedelta(days=4)           # Friday
        mstr, fstr = wk_start.strftime("%Y-%m-%d"), fri_date.strftime("%Y-%m-%d")

        # 1️⃣  Monday stock closes
        stocks = pd.read_sql(f"""
            SELECT symbol, close AS stock_close
            FROM   stocks
            WHERE  quote_date = '{mstr}'
        """, stk_con)

        # 2️⃣  Monday put quotes that EXPIRE that Friday
        puts = pd.read_sql(f"""
            SELECT *
            FROM   options
            WHERE  quote_date = '{mstr}'
              AND  type       = 'put'
              AND  expiration = '{fstr}'
        """, opt_con)
        if puts.empty or stocks.empty:
            continue

        # price columns are in cents → convert to dollars
        for col in ("bid", "ask", "last", "mid"):
            if col in puts.columns:
                puts[col] = puts[col] / 100.0

        # 3️⃣  merge & clean
        merged = puts.merge(stocks, left_on="underlying", right_on="symbol")
        merged = merged[merged["underlying"].isin(SP500_SET)].drop_duplicates(
            subset=["underlying", "strike", "bid"]
        )
        if merged.empty:
            continue

        # 4️⃣  volatility lookup (last `lookback` closes)
        sig = {}
        for sym in merged["underlying"].unique():
            hist = pd.read_sql(f"""
                SELECT close FROM stocks
                WHERE  symbol = '{sym}'
                  AND  quote_date <= '{mstr}'
                ORDER BY quote_date DESC
                LIMIT  {lookback}
            """, stk_con)
            if len(hist) > 1:
                ret = hist["close"].pct_change().dropna()
                sig[sym] = ret.std() * math.sqrt(252)

        # 5️⃣  score puts
        days_to_exp = (fri_date - wk_start).days
        T_week = days_to_exp / 365.0                      # same for all rows

        for _, row in merged.iterrows():
            S, K, bid = row["stock_close"], row["strike"], row["bid"]
            sigma = sig.get(row["underlying"])
            if not sigma or bid <= 0:
                continue
            model = _bs_put(S, K, T_week, r, sigma)
            edge  = model - bid
            all_rows.append({
                "week"        : mstr,
                "ticker"      : row["underlying"],
                "strike"      : K,
                "stock_close" : S,
                "market_bid"  : bid,
                "bs_price"    : round(model, 2),
                "edge"        : round(edge, 2),
                "sigma"       : round(sigma, 4),
                "expiration"  : row["expiration"]
            })

    opt_con.close(); stk_con.close()

    picks = (pd.DataFrame(all_rows)
             .sort_values(["week", "edge"], ascending=[True, False])
             .groupby("week")
             .head(top_n)
             .reset_index(drop=True))
    return picks

# ──────────────────────────────────────────────────────────────
#  2. Back-test: buy Monday, exit Friday
# ──────────────────────────────────────────────────────────────
def backtest_puts(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()

    opt_con, stk_con = _connect()
    out = []
    for _, row in picks.iterrows():
        mon  = datetime.strptime(row["week"], "%Y-%m-%d")
        fri  = mon + timedelta(days=4)
        fstr = fri.strftime("%Y-%m-%d")

        fc = pd.read_sql(f"""
            SELECT close FROM stocks
            WHERE symbol='{row['ticker']}' AND quote_date='{fstr}'
            LIMIT 1
        """, stk_con)
        if fc.empty:
            continue

        S_fri  = fc.iat[0, 0]
        payoff = max(0, row["strike"] - S_fri)
        pnl    = payoff - row["market_bid"]

        out.append({
            **row,
            "friday"  : fstr,
            "S_fri"   : S_fri,
            "payoff"  : payoff,
            "PnL"     : pnl,
            "assigned": S_fri < row["strike"]
        })

    opt_con.close(); stk_con.close()
    return pd.DataFrame(out)


# ──────────────────────────────────────────────────────────────
#  NEW: capital-compounding PnL plot
# ──────────────────────────────────────────────────────────────
def plot_weekly_pnl_cmpd(trades: pd.DataFrame,
                    *,
                    starting_capital=500_000,
                    picks_per_week=20):
    """
    Compounds gains/losses each week and plots the account value.

    Parameters
    ----------
    trades : DataFrame
        Must be the result of backtest_puts(), one row per option with
        columns ['week', 'market_bid', 'PnL'] at minimum.
    starting_capital : float
        Initial cash in dollars.
    picks_per_week : int
        How many puts you take each Monday (used for equal-sized slices).
    """
    if trades.empty:
        print("No trades to plot.")
        return trades

    # Ensure week order
    trades_sorted = trades.sort_values("week")
    weeks = trades_sorted["week"].unique()

    ledger = []
    capital = starting_capital

    for wk in weeks:
        wk_trades = trades_sorted[trades_sorted["week"] == wk]

        # Equal slice of current capital
        alloc = capital / picks_per_week
        week_pnl = 0

        for _, r in wk_trades.iterrows():
            cost_per_contract = r["market_bid"] * 100
            if cost_per_contract <= 0:
                continue
            contracts = math.floor(alloc / cost_per_contract)
            week_pnl += r["PnL"] * 100 * contracts

        ledger.append({
            "week"            : wk,
            "starting_capital": capital,
            "weekly_PnL"      : week_pnl,
            "ending_capital"  : capital + week_pnl
        })
        capital += week_pnl     # compound

    ledger_df = pd.DataFrame(ledger)

    # ---- plot running account value ----
    plt.figure(figsize=(8,4))
    plt.plot(ledger_df["week"], ledger_df["ending_capital"], marker="o")
    plt.title("Account Value with Weekly Compounding")
    plt.xlabel("Week Start")
    plt.ylabel("Capital ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nFinal capital: ${capital:,.2f}")
    return ledger_df


# ──────────────────────────────────────────────────────────────
#  3. Plot weekly PnL with a fixed capital pool
# ──────────────────────────────────────────────────────────────
def plot_weekly_pnl(trades: pd.DataFrame, *,
                    capital=500_000, picks_per_week=20):
    if trades.empty:
        print("No trades to plot.")
        return pd.DataFrame()

    alloc = capital / picks_per_week
    pnl_rows = []

    for wk, grp in trades.groupby("week"):
        total = 0
        for _, r in grp.iterrows():
            cost      = r["market_bid"] * 100
            contracts = math.floor(alloc / cost) if cost > 0 else 0
            total    += r["PnL"] * 100 * contracts
        pnl_rows.append({"week": wk, "weekly_PnL": total})

    pnl_df = pd.DataFrame(pnl_rows).sort_values("week")

    plt.figure(figsize=(8, 4))
    plt.plot(pnl_df["week"], pnl_df["weekly_PnL"], marker="o")
    plt.xticks(rotation=45)
    plt.title(f"Weekly PnL – ${capital:,.0f} equally across "
              f"{picks_per_week} puts")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pnl_df