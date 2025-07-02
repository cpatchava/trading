import pandas as pd
import os

def fetch_and_save_sp500_symbols(path="../sql-database/sp500_symbols.txt"):
    """
    Fetches the current list of S&P 500 companies from Wikipedia
    and saves their symbols to a local file for downstream use.
    """
    print("📥 Fetching S&P 500 symbols from Wikipedia...")
    tbl = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    symbols = tbl["Symbol"].str.replace(".", "-", regex=False).dropna().unique()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for symbol in symbols:
            f.write(symbol + "\n")

    print(f"✅ Saved {len(symbols)} symbols to {path}")

if __name__ == "__main__":
    fetch_and_save_sp500_symbols()

