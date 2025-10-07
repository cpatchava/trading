# Project README

This repository contains a set of Python scripts and SQLite databases for ingesting, enriching, and modeling equity and options data—culminating in Black–Scholes option pricing. Clone this repo, install the dependencies, and run each script in order to take raw CSVs all the way to model prices.

---

## 📦 Prerequisites

- **Python** 3.8+ (tested on 3.13)  
- **SQLite** (bundled with Python)  
- It’s recommended to create and activate a virtual environment:  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Install Python dependencies:  
  ```bash
  pip install pandas numpy sqlalchemy argparse
  ```

---

## 🔧 Scripts

### 1. `load_data.py`

**What it does**  
Walks a directory of raw CSVs, parses filenames for quote dates, and loads them into two SQLite tables, `stocks` and `options`.

**Run:**
```bash
python load_data.py \
  --data-root path/to/your/csv/folder \
  --options-db ../sql-database/options_data.db \
  --stocks-db  ../sql-database/stocks_data.db
```

---

### 2. `fetch_and_save_sp500_symbols.py`

**What it does**  
Scrapes the current list of S&P 500 tickers from Wikipedia, normalizes them, and writes them to a plain-text file for downstream use.

**Run:**
```bash
python fetch_and_save_sp500_symbols.py \
  --path ../sql-database/sp500_symbols.txt
```

---

### 3. `stocks_enrichment.py`

**What it does**  
- Loads the raw `stocks` table and the S&P 500 ticker list  
- Flags each record with `is_sp500 = 1|0`  
- Computes annualized volatility over rolling windows of 1 week, 2 weeks, 1 month, and 3 months (best-effort by default)  
- Writes the enriched output to a new table `stocks_enriched`

**Run:**
```bash
python stocks_enrichment.py \
  --stocks-db ../sql-database/stocks_data.db \
  --sp500-file ../sql-database/sp500_symbols.txt
```

---

### 4. `options_enrichment.py`

**What it does**  
- Reads your raw `options` table  
- Calculates `num_days` until expiration for each contract  
- Writes everything out to a new table `enriched_options`

**Run:**
```bash
python options_enrichment.py \
  --options-db ../sql-database/options_data.db
```

---

### 5. `black_scholes_modeling.py`

**What it does**  
- Loads `enriched_options` (with `num_days`) and `stocks_enriched` (with volatilities and optional `dividend_yield`)  
- Selects the correct volatility bucket based on days-to-expiry (1 wk, 2 wks, 1 mo, 3 mo)  
- Computes \(T\) in years, pulls in a continuous annual risk-free rate (`r`), and applies the Black–Scholes formula (calls & puts, with continuous dividend yield \(q\))  
- Updates `enriched_options` with a new `black_scholes_model_price` column

**Run:**
```bash
python black_scholes_modeling.py \
  --options-db      ../sql-database/options_data.db \
  --stocks-db       ../sql-database/stocks_data.db \
  --risk-free-rate  0.015
```

---

## ✅ Tests

There’s one example pytest for `stocks_enrichment.py`. To run it:
```bash
pytest test_stocks_enrichment.py
```

---

That’s it! Follow the steps above in order to load raw data, enrich it, and compute theoretical option prices using the Black–Scholes model.
```
