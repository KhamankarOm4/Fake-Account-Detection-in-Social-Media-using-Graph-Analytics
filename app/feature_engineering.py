"""
feature_engineering.py
-----------------------
Computes per-node graph features for fake account detection:
  - In-degree / Out-degree
  - Degree centrality
  - PageRank (optimized with max_iter + tol tuning)
  - Clustering coefficient (on undirected projection)
  - Approximate betweenness centrality (k-sample)

All computations are limited to top_n nodes for performance on large graphs.
"""

import networkx as nx
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_features(
    G: nx.DiGraph,
    top_nodes: list,
    pagerank_alpha: float = 0.85,
    pagerank_max_iter: int = 100,
    betweenness_k: int = 200,
) -> pd.DataFrame:
    """
    Compute graph features for the given list of nodes.

    Parameters
    ----------
    G : nx.DiGraph
        The directed social network graph.
    top_nodes : list
        Nodes to compute features for (pre-filtered for performance).
    pagerank_alpha : float
        Damping factor for PageRank.
    pagerank_max_iter : int
        Max iterations for PageRank convergence.
    betweenness_k : int
        Number of sample pivots for approximate betweenness centrality.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per node.
    """
    node_set = set(top_nodes)
    subgraph = G.subgraph(top_nodes)  # lightweight view — no copy

    logger.info(f"Computing features for {len(top_nodes):,} nodes...")

    # ── 1. Degree features ──────────────────────────────────────────────────
    in_degree = dict(subgraph.in_degree())
    out_degree = dict(subgraph.out_degree())
    logger.debug("  ✓ In/Out degree computed")

    # ── 2. Degree centrality ────────────────────────────────────────────────
    degree_centrality = nx.degree_centrality(subgraph)
    logger.debug("  ✓ Degree centrality computed")

    # ── 3. PageRank ─────────────────────────────────────────────────────────
    try:
        pagerank = nx.pagerank(
            subgraph,
            alpha=pagerank_alpha,
            max_iter=pagerank_max_iter,
            tol=1e-4,        # relaxed tolerance for speed
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("  PageRank did not converge — using degree-normalized fallback")
        total_edges = subgraph.number_of_edges() or 1
        pagerank = {n: subgraph.in_degree(n) / total_edges for n in subgraph.nodes()}
    logger.debug("  ✓ PageRank computed")

    # ── 4. Clustering coefficient (undirected projection) ───────────────────
    G_undirected = subgraph.to_undirected()
    clustering = nx.clustering(G_undirected)
    logger.debug("  ✓ Clustering coefficient computed")

    # ── 5. Approximate betweenness centrality (k-sample) ────────────────────
    k_actual = min(betweenness_k, len(top_nodes))
    try:
        betweenness = nx.betweenness_centrality(subgraph, k=k_actual, normalized=True)
    except Exception as e:
        logger.warning(f"  Betweenness centrality failed ({e}) — defaulting to 0")
        betweenness = {n: 0.0 for n in subgraph.nodes()}
    logger.debug("  ✓ Approximate betweenness centrality computed")

    # ── 6. Assemble feature DataFrame ───────────────────────────────────────
    records = []
    for node in top_nodes:
        records.append({
            "node": node,
            "in_degree":          in_degree.get(node, 0),
            "out_degree":         out_degree.get(node, 0),
            "degree_centrality":  round(degree_centrality.get(node, 0.0), 6),
            "pagerank":           round(pagerank.get(node, 0.0), 8),
            "clustering_coeff":   round(clustering.get(node, 0.0), 6),
            "betweenness":        round(betweenness.get(node, 0.0), 8),
        })

    df_features = pd.DataFrame(records)
    logger.info(f"Feature matrix shape: {df_features.shape}")
    return df_features
