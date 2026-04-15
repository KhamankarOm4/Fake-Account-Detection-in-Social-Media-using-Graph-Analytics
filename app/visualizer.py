"""
visualizer.py
-------------
Optional PyVis graph visualization endpoint.
Renders an interactive HTML graph with fake nodes highlighted in red
and real nodes in green/teal.
Designed to work with a subgraph of manageable size (≤ 500 nodes).
"""

import os
import logging
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# Output directory for generated HTML files
VIZ_DIR = "/tmp/viz"


def _ensure_viz_dir():
    os.makedirs(VIZ_DIR, exist_ok=True)


def generate_pyvis_graph(
    G: nx.DiGraph,
    df_results: pd.DataFrame,
    max_nodes: int = 300,
    output_filename: str = "graph.html",
) -> str:
    """
    Generate an interactive HTML graph visualization using PyVis.
    Fake nodes are shown in red, real nodes in teal.

    Parameters
    ----------
    G : nx.DiGraph
        The full directed graph.
    df_results : pd.DataFrame
        Detection results with 'node', 'is_fake', 'ml_probability', 'reason'.
    max_nodes : int
        Cap on number of nodes to render (for browser performance).
    output_filename : str
        Name of the output HTML file.

    Returns
    -------
    str : Absolute path to the generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis is not installed. Add 'pyvis' to requirements.txt.")

    _ensure_viz_dir()

    # ── Build label lookup ──────────────────────────────────────────────────
    label_map = dict(zip(df_results["node"], df_results["is_fake"]))
    prob_map  = dict(zip(df_results["node"], df_results.get("ml_probability", [0]*len(df_results))))
    reason_map = dict(zip(df_results["node"], df_results["reason"]))

    # ── Sample a subgraph for visualization ────────────────────────────────
    all_nodes = list(df_results["node"])[:max_nodes]
    subgraph = G.subgraph(all_nodes)

    logger.info(f"Generating PyVis graph for {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges...")

    # ── Configure PyVis Network ─────────────────────────────────────────────
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#0d1117",        # dark background
        font_color="#e6edf3",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)

    for node in subgraph.nodes():
        is_fake = label_map.get(node, 0)
        prob    = prob_map.get(node, 0.0)
        reason  = reason_map.get(node, "")

        color   = "#ef4444" if is_fake else "#22d3ee"   # red vs cyan
        border  = "#b91c1c" if is_fake else "#0891b2"
        size    = 20 + int(prob * 30) if is_fake else 12
        title   = (
            f"<b>Node {node}</b><br>"
            f"Status: {'🚨 FAKE' if is_fake else '✅ Real'}<br>"
            f"Confidence: {prob:.2%}<br>"
            f"<i>{reason}</i>"
        )

        net.add_node(
            node,
            label=str(node),
            color={"background": color, "border": border},
            size=size,
            title=title,
            font={"color": "#ffffff", "size": 10},
        )

    for src, tgt in subgraph.edges():
        src_fake = label_map.get(src, 0)
        edge_color = "#ef444440" if src_fake else "#22d3ee20"
        net.add_edge(src, tgt, color=edge_color, width=0.5, arrows="to")

    # ── Export ──────────────────────────────────────────────────────────────
    output_path = os.path.join(VIZ_DIR, output_filename)
    net.save_graph(output_path)
    logger.info(f"PyVis graph saved to {output_path}")
    return output_path


def build_dashboard_summary(df_results: pd.DataFrame) -> dict:
    """
    Build a dashboard-ready summary dict from detection results.
    Useful for frontend consumption without a full graph render.
    """
    total = len(df_results)
    fake_df  = df_results[df_results["is_fake"] == 1]
    real_df  = df_results[df_results["is_fake"] == 0]

    return {
        "total_analyzed": total,
        "fake_count": len(fake_df),
        "real_count": len(real_df),
        "fake_percentage": round(100 * len(fake_df) / max(total, 1), 2),
        "avg_pagerank_fake": round(fake_df["pagerank"].mean(), 8) if len(fake_df) else 0,
        "avg_pagerank_real": round(real_df["pagerank"].mean(), 8) if len(real_df) else 0,
        "avg_clustering_fake": round(fake_df["clustering_coeff"].mean(), 6) if len(fake_df) else 0,
        "avg_clustering_real": round(real_df["clustering_coeff"].mean(), 6) if len(real_df) else 0,
        "top_fake_nodes": (
            fake_df.nlargest(10, "ml_probability")[["node", "ml_probability", "out_degree", "in_degree", "reason"]]
            .assign(node=lambda d: d["node"].astype(int))
            .to_dict(orient="records")
        ) if len(fake_df) else [],
    }
