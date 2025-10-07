import sqlite3
import pandas as pd
import numpy as np
import pytest
import tempfile
import os
from stocks_enrichment import load_sp500_symbols, enrich_stocks_table, WINDOWS

@pytest.fixture
def temp_db_and_data(tmp_path):
    # Create a temporary SQLite DB
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute('''
        CREATE TABLE stocks (
            symbol TEXT,
            open REAL, high REAL, low REAL,
            close REAL, volume INTEGER,
            adjust_close REAL, quote_date TEXT
        )
    ''')
    # Insert 10 business days of data for two symbols:
    # AAA with constant price=100, BBB with increasing prices 10,20,...,100
    dates = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='B')
    records = []
    for i, date in enumerate(dates):
        records.append(('AAA', 100, 100, 100, 100, 1000, 100, date.strftime('%Y-%m-%d')))
        price = 10 * (i + 1)
        records.append(('BBB', price, price, price, price, 1000, price, date.strftime('%Y-%m-%d')))
    conn.executemany('INSERT INTO stocks VALUES (?,?,?,?,?,?,?,?)', records)
    conn.commit()
    conn.close()

    # Create a temporary S&P 500 symbols file containing only "AAA"
    sp500_file = tmp_path / "sp500_symbols.txt"
    sp500_file.write_text("AAA\n")

    return str(db_file), str(sp500_file)

def test_is_sp500_flag_and_volatility(temp_db_and_data):
    db_file, sp500_path = temp_db_and_data

    # 1. Load and verify SP500 symbols
    symbols = load_sp500_symbols(sp500_path)
    assert symbols == {"AAA"}

    # 2. Run enrichment
    enrich_stocks_table(db_file, symbols)

    # 3. Read enriched data
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM stocks_enriched", conn)
    conn.close()

    # 4. Check is_sp500 flag
    aaa_flag = df[df['symbol']=='AAA']['is_sp500'].unique().tolist()
    bbb_flag = df[df['symbol']=='BBB']['is_sp500'].unique().tolist()
    assert aaa_flag == [1]
    assert bbb_flag == [0]

    # 5. Verify volatility columns
    for col, window in WINDOWS.items():
        assert col in df.columns

        # AAA: constant price → volatility zero after the first valid window
        aaa_vol = df[df['symbol']=='AAA'][col].dropna().iloc[window-1:]
        assert np.allclose(aaa_vol, 0.0)

        # BBB: increasing price → at least one positive volatility
        bbb_vol = df[df['symbol']=='BBB'][col].dropna()
        assert (bbb_vol > 0).any()
