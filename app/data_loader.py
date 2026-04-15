"""
data_loader.py
--------------
Handles loading of edge-list datasets efficiently using pandas with chunking.
Supports partial loading and memory-optimized dtypes for large graphs (1M+ edges).
"""

import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def load_edgelist(
    filepath: str,
    chunksize: int = 100_000,
    max_rows: int = None,
    sep: str = " ",
) -> pd.DataFrame:
    """
    Load an edge-list file (source, target) in memory-efficient chunks.

    Parameters
    ----------
    filepath : str
        Path to the edge-list text/CSV file.
    chunksize : int
        Number of rows to read per chunk (default 100k).
    max_rows : int | None
        If set, stop loading after this many rows (partial load).
    sep : str
        Column separator in the file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source', 'target'] using uint32 dtypes
        to minimise memory usage.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    chunks = []
    total_loaded = 0

    logger.info(f"Loading dataset from {filepath} (chunksize={chunksize}, max_rows={max_rows})")

    try:
        reader = pd.read_csv(
            filepath,
            sep=sep,
            names=["source", "target"],
            dtype={"source": "int32", "target": "int32"},
            comment="#",          # skip comment lines
            on_bad_lines="skip",  # skip malformed rows
            chunksize=chunksize,
            engine="c",           # fastest pandas engine
        )

        for chunk in reader:
            # Drop self-loops & nulls
            chunk = chunk.dropna()
            chunk = chunk[chunk["source"] != chunk["target"]]
            chunks.append(chunk)
            total_loaded += len(chunk)
            logger.debug(f"  Loaded chunk: {len(chunk)} rows (total so far: {total_loaded})")

            if max_rows and total_loaded >= max_rows:
                logger.info(f"Partial load limit reached ({max_rows} rows).")
                break

    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        raise

    if not chunks:
        raise ValueError("No valid data loaded from dataset.")

    df = pd.concat(chunks, ignore_index=True)

    # Trim to exact max_rows if needed
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]

    logger.info(f"Dataset loaded: {len(df):,} edges, memory ≈ {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def get_dataset_stats(filepath: str, sep: str = " ") -> dict:
    """
    Quickly count total lines in a file without loading it into memory.
    Useful for large dataset preview before loading.
    """
    if not os.path.exists(filepath):
        return {"error": "File not found", "path": filepath}

    line_count = 0
    size_bytes = os.path.getsize(filepath)

    with open(filepath, "r") as f:
        for _ in f:
            line_count += 1

    return {
        "path": filepath,
        "total_lines": line_count,
        "file_size_mb": round(size_bytes / 1e6, 2),
    }
