# Options Analytics Pipeline

This project implements a weekly put-selling strategy analysis using the Black–Scholes model. The pipeline consists of four main steps, each encapsulated in a Python script:

1. **Load raw CSV data** into SQLite → [`data-loader.py`](data-loader.py)
2. **Generate model inputs** (`bs_models`) → [`scripts/01_create_bs_models.py`](scripts/01_create_bs_models.py)
3. **Compute BS theoretical prices** (`bs_scores`) → [`scripts/02_generate_bs_scores.py`](scripts/02_generate_bs_scores.py)
4. **Rank weekly underpriced puts** → [`scripts/03_weekly_put_rank.py`](scripts/03_weekly_put_rank.py)

---

## Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure these files exist in the `data/` folder:
  - `options_data.db` (raw options table)
  - `stocks_data.db`  (raw stocks table)
  - `sp500_symbols.txt` (one symbol per line)

---

## Step 0: Load Raw CSV Data

Use `data-loader.py` to ingest all `*options.csv` and `*stocks.csv` files under `data/` into your SQLite databases.

```bash
python data-loader.py
```

This script will:

- Create (if missing) `options` and `stocks` tables in:
  - `sql-database/options_data.db`
  - `sql-database/stocks_data.db`
- Walk the `data/` directory for `*options.csv` and `*stocks.csv` files.
- Append each CSV into its corresponding table, automatically adding a `quote_date` from the filename.

**Key snippet** from [`data-loader.py`](data-loader.py):

```python
# Derive quote_date from filename
date_str = os.path.basename(csv_path).split("options")[0].split("stocks")[0].strip()
if extra_columns:
    df["quote_date"] = date_str
```

---

## Step 1: Generate BS Models

Compute the raw model inputs—days-to-expiry, mid-prices, weekday, historical volatility, and S&P 500 flag—then write to `bs_models`.

```bash
python scripts/01_create_bs_models.py \
  --db-path        data/options_data.db \
  --db-path-stocks data/stocks_data.db \
  --sp500-path     data/sp500_symbols.txt \
  --hist-window    20
```

**Key snippet** from [`scripts/01_create_bs_models.py`](scripts/01_create_bs_models.py):

```python
# days to expiry and mid-price
opts['days_to_expiry'] = (opts.expiration - opts.quote_date).dt.days
opts['mid_price']      = (opts.bid + opts.ask) / 2.0

# 20-day annualized volatility
opts['hist_vol'] = (
    stocks_sorted.groupby('symbol')['log_ret']
        .rolling(window=hist_window).std()
        .reset_index(level=0, drop=True)
    * np.sqrt(252)
)
```

---

## Step 2: Generate BS Scores

Merge `bs_models` with underlying prices, compute Black–Scholes theoretical prices and the mid-model spread, writing to `bs_scores`.

```bash
python scripts/02_generate_bs_scores.py \
  --options-db     data/options_data.db \
  --stocks-db      data/stocks_data.db \
  --risk-free-rate 0.01 \
  --dividend-yield 0.0
```

**Key snippet** from [`scripts/02_generate_bs_scores.py`](scripts/02_generate_bs_scores.py):

```python
# Black–Scholes formula
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
# put vs call pricing
bs_price = np.where(
    merged['type']=='call',
    S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2),
    K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
)
```

---

## Step 3: Rank Weekly Underpriced Puts

Filter for puts, group by calendar week, select the 20 most underpriced (`price_diff` most negative), include liquidity (`open_interest`, `volume`), and write to `weekly_put_rank`.

```bash
python scripts/03_weekly_put_rank.py \
  --options-db    data/options_data.db \
  --output-table  weekly_put_rank
```

**Key snippet** from [`scripts/03_weekly_put_rank.py`](scripts/03_weekly_put_rank.py):

```python
# ISO week and ranking
f['iso_week']  = f['quote_date'].dt.strftime('%Y-%W')
df_sorted      = df.sort_values(['iso_week','price_diff'])
df_top20       = df_sorted.groupby('iso_week').head(20)
```

---

You’re now set to inspect `weekly_put_rank` in `data/options_data.db` to review top put-selling opportunities week-over-week.

