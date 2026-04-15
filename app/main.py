"""
main.py
-------
Flask API server for Fake Account Detection in Social Media using Graph Analytics.

Endpoints
---------
  GET /              → Health check + system info
  GET /stats         → Graph-level statistics
  GET /analyze       → Full pipeline: features + detection + ML metrics
  GET /fake-users    → Paginated list of detected fake accounts
  GET /visualize     → Generate interactive PyVis HTML graph (bonus)
  GET /dashboard     → Dashboard-ready summary JSON
  GET /cache-info    → Cache diagnostics

Performance Strategy
--------------------
  - Graph and features are computed ONCE and cached in memory (1-hour TTL).
  - All heavy ops are limited to top_n nodes (configurable via ?top_n=...).
  - Chunked loading prevents OOM on 1M+ edge datasets.
"""

import os
import logging
import time
from flask import Flask, jsonify, request, send_file, abort

# ── Local modules ────────────────────────────────────────────────────────────
from data_loader      import load_edgelist, get_dataset_stats
from graph_builder    import build_graph, get_top_nodes_by_degree, graph_summary
from feature_engineering import compute_features
from detector         import rule_based_detection, ml_detection, combine_labels
from visualizer       import generate_pyvis_graph, build_dashboard_summary
import cache

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)

DATASET_PATH  = os.environ.get("DATASET_PATH", "/data/twitter_combined.txt")
DATASET_SEP   = os.environ.get("DATASET_SEP",  " ")
MAX_ROWS      = int(os.environ.get("MAX_ROWS",  "500000"))   # partial load cap
TOP_N         = int(os.environ.get("TOP_N",     "5000"))     # nodes to analyze
CHUNKSIZE     = int(os.environ.get("CHUNKSIZE", "100000"))

# ── Cache keys ────────────────────────────────────────────────────────────────
CACHE_GRAPH    = "graph"
CACHE_FEATURES = "features"
CACHE_RESULTS  = "results"
CACHE_METRICS  = "ml_metrics"


# ════════════════════════════════════════════════════════════════════════════
# Bootstrap: lazy-load graph on first request
# ════════════════════════════════════════════════════════════════════════════

def _get_or_build_graph():
    """Return cached graph, building it if necessary."""
    G = cache.get_cache(CACHE_GRAPH)
    if G is None:
        logger.info("Graph cache miss — loading dataset and building graph...")
        df_edges = load_edgelist(DATASET_PATH, chunksize=CHUNKSIZE, max_rows=MAX_ROWS, sep=DATASET_SEP)
        G = build_graph(df_edges)
        cache.set_cache(CACHE_GRAPH, G, ttl=3600)
    return G


def _get_or_compute_results(top_n: int = TOP_N):
    """Return cached full detection results, computing if necessary."""
    results_key = f"{CACHE_RESULTS}_{top_n}"
    metrics_key = f"{CACHE_METRICS}_{top_n}"

    cached_results = cache.get_cache(results_key)
    cached_metrics = cache.get_cache(metrics_key)

    if cached_results is not None and cached_metrics is not None:
        return cached_results, cached_metrics

    # Build graph
    G = _get_or_build_graph()

    # Select top-N nodes
    top_nodes = get_top_nodes_by_degree(G, top_n=top_n)

    # Compute features
    df_features = compute_features(G, top_nodes)

    # Rule-based pass
    df_features = rule_based_detection(df_features)

    # ML pass
    df_features, metrics = ml_detection(df_features)

    # Combine labels
    df_features = combine_labels(df_features)

    cache.set_cache(results_key, df_features, ttl=3600)
    cache.set_cache(metrics_key, metrics,     ttl=3600)

    return df_features, metrics


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health_check():
    """
    GET /
    -----
    Health check endpoint. Returns service status and basic config.
    """
    dataset_exists = os.path.exists(DATASET_PATH)
    return jsonify({
        "status": "ok",
        "service": "Fake Account Detection API",
        "version": "2.0.0",
        "dataset_path": DATASET_PATH,
        "dataset_loaded": dataset_exists,
        "config": {
            "max_rows":  MAX_ROWS,
            "top_n":     TOP_N,
            "chunksize": CHUNKSIZE,
        },
        "cache": cache.cache_info(),
        "endpoints": ["/", "/ui", "/stats", "/analyze", "/fake-users", "/visualize", "/dashboard", "/cache-info"],
    }), 200

@app.route("/ui", methods=["GET"])
def ui():
    """
    GET /ui
    -------
    Serves the beautiful vanilla JS/CSS frontend dashboard.
    """
    return app.send_static_file("index.html")


@app.route("/stats", methods=["GET"])
def get_stats():
    """
    GET /stats
    ----------
    Return structural statistics of the loaded graph.
    Also includes raw dataset file info.
    """
    t0 = time.time()
    try:
        G = _get_or_build_graph()
        summary = graph_summary(G)
        file_info = get_dataset_stats(DATASET_PATH)
        return jsonify({
            "status": "ok",
            "graph_stats": summary,
            "dataset_file": file_info,
            "elapsed_seconds": round(time.time() - t0, 3),
        }), 200
    except Exception as e:
        logger.exception("Error in /stats")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/analyze", methods=["GET"])
def analyze():
    """
    GET /analyze?top_n=<int>
    ------------------------
    Run the full pipeline:
      1. Build / load graph
      2. Compute graph features for top-N nodes
      3. Rule-based detection
      4. ML-based detection (RandomForest)
      5. Return results + ML evaluation metrics

    Query Params
    ------------
    top_n : int  (default = TOP_N env var, default 5000)
    """
    t0 = time.time()
    try:
        top_n = int(request.args.get("top_n", TOP_N))
        top_n = max(100, min(top_n, 50_000))   # clamp to safe range

        df_results, metrics = _get_or_compute_results(top_n=top_n)

        # Convert to JSON-safe records
        records = df_results[[
            "node", "in_degree", "out_degree",
            "degree_centrality", "pagerank", "clustering_coeff", "betweenness",
            "rule_label", "ml_label", "ml_probability", "is_fake", "reason",
        ]].copy()

        records["node"] = records["node"].astype(int)

        # Sanitise: replace inf/nan
        records = records.fillna(0)

        # Summary counts
        total   = len(records)
        n_fake  = int(records["is_fake"].sum())
        n_real  = total - n_fake

        return jsonify({
            "status": "ok",
            "summary": {
                "total_analyzed": total,
                "fake_count":     n_fake,
                "real_count":     n_real,
                "fake_percentage": round(100 * n_fake / max(total, 1), 2),
            },
            "ml_metrics": metrics,
            "results":    records.to_dict(orient="records"),
            "elapsed_seconds": round(time.time() - t0, 3),
        }), 200

    except Exception as e:
        logger.exception("Error in /analyze")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/fake-users", methods=["GET"])
def fake_users():
    """
    GET /fake-users?top_n=<int>&page=<int>&page_size=<int>&sort_by=<field>
    -----------------------------------------------------------------------
    Return a paginated, sorted list of detected fake accounts with
    their graph features and explainability reasons.

    Query Params
    ------------
    top_n     : int  (default 5000)
    page      : int  (default 1)
    page_size : int  (default 50, max 500)
    sort_by   : str  (ml_probability | out_degree | pagerank — default: ml_probability)
    """
    t0 = time.time()
    try:
        top_n     = int(request.args.get("top_n",     TOP_N))
        page      = max(1, int(request.args.get("page",      1)))
        page_size = min(500, max(1, int(request.args.get("page_size", 50))))
        sort_by   = request.args.get("sort_by", "ml_probability")

        allowed_sort = {"ml_probability", "out_degree", "in_degree", "pagerank", "betweenness"}
        if sort_by not in allowed_sort:
            sort_by = "ml_probability"

        df_results, _ = _get_or_compute_results(top_n=top_n)

        fakes = df_results[df_results["is_fake"] == 1].copy()
        fakes = fakes.sort_values(sort_by, ascending=False)

        total_fakes = len(fakes)
        start = (page - 1) * page_size
        end   = start + page_size
        page_data = fakes.iloc[start:end]

        out_cols = [
            "node", "in_degree", "out_degree",
            "clustering_coeff", "pagerank", "ml_probability",
            "rule_label", "ml_label", "reason",
        ]
        records = page_data[out_cols].copy()
        records["node"] = records["node"].astype(int)
        records = records.fillna(0)

        return jsonify({
            "status":       "ok",
            "total_fakes":  total_fakes,
            "page":         page,
            "page_size":    page_size,
            "total_pages":  -(-total_fakes // page_size),   # ceiling division
            "sort_by":      sort_by,
            "fake_accounts": records.to_dict(orient="records"),
            "elapsed_seconds": round(time.time() - t0, 3),
        }), 200

    except Exception as e:
        logger.exception("Error in /fake-users")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    GET /dashboard?top_n=<int>
    --------------------------
    Returns a compact, dashboard-ready JSON summary of detection results.
    Suitable for feeding charts and KPI cards on a frontend.
    """
    t0 = time.time()
    try:
        top_n = int(request.args.get("top_n", TOP_N))
        df_results, metrics = _get_or_compute_results(top_n=top_n)
        G = _get_or_build_graph()

        graph_stats = graph_summary(G)
        summary = build_dashboard_summary(df_results)

        return jsonify({
            "status":       "ok",
            "graph_stats":  graph_stats,
            "detection":    summary,
            "ml_metrics": {
                k: v for k, v in metrics.items() if k != "report"   # skip verbose report
            },
            "elapsed_seconds": round(time.time() - t0, 3),
        }), 200

    except Exception as e:
        logger.exception("Error in /dashboard")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/visualize", methods=["GET"])
def visualize():
    """
    GET /visualize?top_n=<int>&max_nodes=<int>
    ------------------------------------------
    Generate an interactive PyVis HTML graph and return it.
    Fake nodes are highlighted in red, real nodes in teal.

    Query Params
    ------------
    top_n     : int  (nodes to analyse, default 5000)
    max_nodes : int  (nodes to render in graph, default 300 for browser perf)
    """
    t0 = time.time()
    try:
        top_n     = int(request.args.get("top_n",     TOP_N))
        max_nodes = int(request.args.get("max_nodes", 300))
        max_nodes = max(50, min(max_nodes, 1000))

        df_results, _ = _get_or_compute_results(top_n=top_n)
        G = _get_or_build_graph()

        html_path = generate_pyvis_graph(G, df_results, max_nodes=max_nodes)
        logger.info(f"/visualize generated in {time.time()-t0:.2f}s → {html_path}")
        return send_file(html_path, mimetype="text/html")

    except ImportError as e:
        return jsonify({"status": "error", "message": str(e), "hint": "Install pyvis: pip install pyvis"}), 501
    except Exception as e:
        logger.exception("Error in /visualize")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/cache-info", methods=["GET"])
def cache_info_endpoint():
    """
    GET /cache-info
    ---------------
    Diagnostic endpoint showing what is currently cached.
    """
    return jsonify({"status": "ok", "cache": cache.cache_info()}), 200


@app.route("/cache-clear", methods=["POST"])
def cache_clear():
    """
    POST /cache-clear
    -----------------
    Invalidate all cached computations (forces full recompute on next request).
    """
    cache.clear_all()
    return jsonify({"status": "ok", "message": "Cache cleared."}), 200


# ════════════════════════════════════════════════════════════════════════════
# Error handlers
# ════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found.", "code": 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error.", "code": 500}), 500


# ════════════════════════════════════════════════════════════════════════════
# Eager Preload (for Gunicorn with --preload)
# ════════════════════════════════════════════════════════════════════════════

if os.environ.get("PRELOAD_DATA", "false").lower() == "true":
    logger.info("PRELOAD_DATA is enabled. Eagerly loading graph and computing features...")
    _get_or_compute_results(top_n=TOP_N)
    logger.info("Preload complete. Gunicorn workers will now fork with shared memory.")

# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Fake Account Detection API on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)