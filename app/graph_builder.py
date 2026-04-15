"""
graph_builder.py
----------------
Constructs a directed NetworkX graph from an edge-list DataFrame.
Includes sampling strategies to handle very large graphs gracefully.
"""

import networkx as nx
import pandas as pd
import logging
import random

logger = logging.getLogger(__name__)


def build_graph(df: pd.DataFrame, sample_nodes: int = None) -> nx.DiGraph:
    """
    Build a directed graph from an edge-list DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'source' and 'target' columns.
    sample_nodes : int | None
        If set, sample a subgraph around this many random seed nodes.
        Use for very large graphs to avoid OOM.

    Returns
    -------
    nx.DiGraph
    """
    logger.info(f"Building directed graph from {len(df):,} edges...")

    if sample_nodes:
        # Sample strategy: pick random seed nodes, take ego-graphs
        unique_nodes = pd.unique(df[["source", "target"]].values.ravel())
        seed_nodes = set(
            random.sample(list(unique_nodes), min(sample_nodes, len(unique_nodes)))
        )
        # Keep only edges where both endpoints are in seed set
        mask = df["source"].isin(seed_nodes) & df["target"].isin(seed_nodes)
        df = df[mask]
        logger.info(f"Sampled graph: {len(df):,} edges after node sampling ({sample_nodes} seeds)")

    G = nx.from_pandas_edgelist(
        df,
        source="source",
        target="target",
        create_using=nx.DiGraph(),
    )

    logger.info(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_top_nodes_by_degree(G: nx.DiGraph, top_n: int = 5000) -> list:
    """
    Return top-N nodes by total degree (in + out).
    Used to limit expensive computations to influential nodes.
    """
    degree_dict = dict(G.degree())  # total degree
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    return sorted_nodes[:top_n]


def graph_summary(G: nx.DiGraph) -> dict:
    """
    Return a lightweight summary dict of graph properties.
    Avoids expensive global metrics.
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "avg_in_degree": round(sum(in_degrees) / max(len(in_degrees), 1), 4),
        "avg_out_degree": round(sum(out_degrees) / max(len(out_degrees), 1), 4),
        "max_in_degree": max(in_degrees, default=0),
        "max_out_degree": max(out_degrees, default=0),
        "density": round(nx.density(G), 8),
    }
