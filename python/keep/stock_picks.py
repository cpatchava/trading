import os, math, sqlite3, pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

DB_DIR   = "../sql-data"        # ←-- update if different
OPT_DB   = os.path.join(DB_DIR, "options_data.db")
STK_DB   = os.path.join(DB_DIR, "stocks_data.db")

# --- Black-Scholes European put --------------------------------------------
def bs_put(S, K, T, r, sigma):
    if min(S, K, T, sigma) <= 0:         # catch bad inputs
        return 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# ---------------------------------------------------------------------------
def weekly_put_screen(start_date="2013-01-07", weeks=4,
                      r=0.05, lookback_days=10, top_n=20):
    """
    Return a DataFrame of the top‐N puts (most under-priced to buy /
    most over-priced to sell) for each Monday–Friday week.
    """
    opt_con  = sqlite3.connect(OPT_DB)
    stk_con  = sqlite3.connect(STK_DB)

    start    = datetime.strptime(start_date, "%Y-%m-%d")
    mondays  = [start + timedelta(weeks=i) for i in range(weeks)]
    T        = 5/365                     # Monday → Friday, ~5 trading days
    rows     = []

    for monday in mondays:
        monday_str = monday.strftime("%Y-%m-%d")

        # 1️⃣  load Monday stock closes
        stk = pd.read_sql(f"""
            SELECT symbol, close
            FROM   stocks
            WHERE  quote_date = '{monday_str}'
        """, stk_con)

        # 2️⃣  load Monday puts
        puts = pd.read_sql(f"""
            SELECT *
            FROM   options
            WHERE  quote_date = '{monday_str}'
            AND    type       = 'put'
        """, opt_con)

        if puts.empty or stk.empty:
            continue

        # 3️⃣  merge to get underlying close
        df = puts.merge(stk, left_on="underlying",
                        right_on="symbol", suffixes=('', '_stk'))

        # 4️⃣  quick volatility lookup (10 previous closes)
        sig_lookup = {}
        for sym in df["underlying"].unique():
            hist = pd.read_sql(f"""
                SELECT close
                FROM   stocks
                WHERE  symbol = '{sym}'
                AND    quote_date <= '{monday_str}'
                ORDER BY quote_date DESC
                LIMIT  {lookback_days}
            """, stk_con)
            if len(hist) > 1:
                ret  = hist["close"].pct_change().dropna()
                sig  = ret.std() * math.sqrt(252)
                sig_lookup[sym] = sig

        # 5️⃣  compute edge and collect
        for _, row in df.iterrows():
            S, K   = row["close_stk"], row["strike"]
            bid    = row["bid"]
            sig    = sig_lookup.get(row["underlying"])
            if sig is None or bid <= 0:
                continue
            model  = bs_put(S, K, T, r, sig)
            edge   = model - bid                # >0 → undervalued / good buy
            rows.append({
                "week"        : monday_str,
                "ticker"      : row["underlying"],
                "strike"      : K,
                "stock_close" : S,
                "market_bid"  : bid,
                "bs_price"    : round(model, 2),
                "edge"        : round(edge, 2),
                "sigma"       : round(sig, 4)
            })

    opt_con.close(); stk_con.close()

    # 6️⃣  rank inside each week & keep top-N
    out = (pd.DataFrame(rows)
           .sort_values(["week", "edge"], ascending=[True, False])
           .groupby("week")
           .head(top_n)
           .reset_index(drop=True))
    return out