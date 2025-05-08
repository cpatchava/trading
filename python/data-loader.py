import os
import sqlite3
import pandas as pd
from glob import glob

# Change this to wherever your data root lives
DATA_ROOT = "../data"

# DB connections
options_conn = sqlite3.connect("options_data.db")
stocks_conn = sqlite3.connect("stocks_data.db")

def create_options_table(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS options (
        contract TEXT, underlying TEXT, expiration TEXT, type TEXT,
        strike REAL, style TEXT, bid REAL, bid_size INTEGER,
        ask REAL, ask_size INTEGER, volume INTEGER, open_interest INTEGER,
        quote_date TEXT, delta REAL, gamma REAL, theta REAL,
        vega REAL, implied_volatility REAL
    )
    ''')
    conn.commit()

def create_stocks_table(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        symbol TEXT, open REAL, high REAL, low REAL,
        close REAL, volume INTEGER, adjust_close REAL,
        quote_date TEXT
    )
    ''')
    conn.commit()

def load_csv_to_db(csv_path, conn, table_name, extra_columns=None):
    try:
        df = pd.read_csv(csv_path)

        # Add quote_date from filename
        date_str = os.path.basename(csv_path).split("options")[0].split("stocks")[0].strip()
        if extra_columns:
            df["quote_date"] = date_str

        df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Loaded: {csv_path}")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")

def main():
    create_options_table(options_conn)
    create_stocks_table(stocks_conn)

    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            full_path = os.path.join(root, file)

            if file.endswith("options.csv"):
                load_csv_to_db(full_path, options_conn, "options", extra_columns=True)
            elif file.endswith("stocks.csv"):
                load_csv_to_db(full_path, stocks_conn, "stocks", extra_columns=True)

    # Close DBs
    options_conn.close()
    stocks_conn.close()
    print("✅ All data loaded.")

if __name__ == "__main__":
    main()

