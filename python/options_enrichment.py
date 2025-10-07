#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Worker: parse expiration & quote_date to datetime, compute num_days.
    """
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["num_days"]   = (df["expiration"] - df["quote_date"]).dt.days
    return df

def main(db_path: str, chunk_size: int, workers: int):
    # 1) Drop any existing enriched_options table
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS enriched_options")
        conn.commit()

    pool = Pool(workers)
    last_rowid = 0
    first = True

    while True:
        # 2) Fetch next chunk by rowid
        with sqlite3.connect(db_path) as reader_conn:
            df_chunk = pd.read_sql_query(
                """
                SELECT rowid AS rowid, *
                FROM options
                WHERE rowid > ?
                ORDER BY rowid
                LIMIT ?
                """,
                reader_conn,
                params=(last_rowid, chunk_size)
            )
        if df_chunk.empty:
            break

        # 3) Advance pointer
        last_rowid = int(df_chunk["rowid"].max())

        # 4) Drop the helper rowid column before processing
        df_chunk = df_chunk.drop(columns=["rowid"])

        # 5) Split into sub‐dataframes and process in parallel
        sub_dfs = np.array_split(df_chunk, workers)
        processed_list = pool.map(process_chunk, sub_dfs)

        # 6) Recombine
        df_enriched = pd.concat(processed_list, ignore_index=True)

        # 7) Write (replace on first chunk, append thereafter)
        with sqlite3.connect(db_path) as writer_conn:
            df_enriched.to_sql(
                "enriched_options",
                writer_conn,
                if_exists="replace" if first else "append",
                index=False
            )

        print(f"↳ Processed rows {last_rowid-chunk_size+1}–{last_rowid}")
        first = False

    pool.close()
    pool.join()
    print("✅ enriched_options has been rebuilt in parallel.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chunked & parallel options enrichment"
    )
    p.add_argument(
        "--options-db",
        default="../sql-database/options_data.db",
        help="Path to your options SQLite DB"
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows to fetch & process per batch"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel worker processes"
    )
    args = p.parse_args()
    main(
        db_path=args.options_db,
        chunk_size=args.chunk_size,
        workers=args.workers
    )
