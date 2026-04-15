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


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

def load_edgelist(
    filepath: str,
    engine: str = "pandas",
    chunksize: int = 100_000,
    max_rows: int = None,
    sep: str = " ",
) -> pd.DataFrame:
    """
    Load an edge-list file (source, target) with toggle between Pandas and Spark.
    
    Parameters
    ----------
    filepath : str
        Path to the edge-list text/CSV file.
    engine : str
        "pandas" (default, chunked) or "spark" (distributed load + limit).
    chunksize : int
        Pandas: rows per chunk (default 100k).
    max_rows : int | None
        Max rows to load (Pandas partial; Spark limit).
    sep : str
        Column separator.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ['source', 'target'] (int32).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    if engine == "spark":
        logger.info(f"Loading with Spark: {filepath} (limit={max_rows or 500000}, sep='{sep}')")
        spark = SparkSession.builder \
            .appName("FakeAccountGraphLoader") \
            .getOrCreate()

        schema = StructType([
            StructField("source", IntegerType(), True),
            StructField("target", IntegerType(), True)
        ])

        df_spark = spark.read \
            .option("sep", sep) \
            .option("header", "false") \
            .option("comment", "#") \
            .option("mode", "DROPMALFORMED") \
            .csv(filepath, schema=schema)

        limit = max_rows or 500000
        df_spark = df_spark.limit(limit).cache()

        df = df_spark.toPandas()

        df_spark.unpersist()
        spark.stop()
        
    else:  # pandas
        logger.info(f"Loading with Pandas: {filepath} (chunksize={chunksize}, max_rows={max_rows})")
        chunks = []
        total_loaded = 0

        try:
            reader = pd.read_csv(
                filepath,
                sep=sep,
                names=["source", "target"],
                dtype={"source": "int32", "target": "int32"},
                comment="#",
                on_bad_lines="skip",
                chunksize=chunksize,
                engine="c",
            )

            for chunk in reader:
                chunk = chunk.dropna()
                chunk = chunk[chunk["source"] != chunk["target"]]
                chunks.append(chunk)
                total_loaded += len(chunk)
                logger.debug(f"  Loaded chunk: {len(chunk)} rows (total: {total_loaded})")

                if max_rows and total_loaded >= max_rows:
                    logger.info(f"Partial load reached ({max_rows} rows).")
                    break

        except Exception as e:
            logger.error(f"Pandas error: {e}")
            raise

        if not chunks:
            raise ValueError("No valid data loaded.")

        df = pd.concat(chunks, ignore_index=True)
        if max_rows and len(df) > max_rows:
            df = df.iloc[:max_rows]

    # Common post-processing
    df = df.dropna()
    df = df[df["source"] != df["target"]]
    df["source"] = df["source"].astype("int32")
    df["target"] = df["target"].astype("int32")

    logger.info(f"Dataset loaded ({engine}): {len(df):,} edges, memory ≈ {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
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
