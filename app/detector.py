"""
detector.py
-----------
Implements two fake-account detection strategies:

  A. Rule-Based:
     - High out-degree + low in-degree ratio
     - Low clustering coefficient
     - Low PageRank

  B. ML-Based (RandomForest):
     - Trains on synthetic labels (from rule-based pass as pseudo-labels)
       when no ground-truth is available.
     - Returns predictions + probabilities + explainability reasons.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Feature columns used for ML ─────────────────────────────────────────────
FEATURE_COLS = [
    "in_degree", "out_degree", "degree_centrality",
    "pagerank", "clustering_coeff", "betweenness",
]


# ═══════════════════════════════════════════════════════════════════════════
# A. RULE-BASED DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def rule_based_detection(
    df: pd.DataFrame,
    out_in_ratio_threshold: float = 10.0,
    min_out_degree: int = 20,
    max_clustering: float = 0.05,
    max_pagerank_percentile: float = 20.0,
) -> pd.DataFrame:
    """
    Flag fake accounts using heuristic rules on graph features.

    Rules (all must hold to flag as fake):
      1. out_degree / (in_degree + 1) > threshold   → follows many, few followers
      2. clustering_coeff < max_clustering            → low community embedding
      3. pagerank < 20th percentile                   → low influence/authority

    Returns df with added columns: 'rule_label', 'rule_reasons'
    """
    df = df.copy()

    pr_threshold = np.percentile(df["pagerank"], max_pagerank_percentile)

    # Compute ratio
    df["out_in_ratio"] = df["out_degree"] / (df["in_degree"] + 1)

    condition_ratio    = (df["out_in_ratio"] > out_in_ratio_threshold) & (df["out_degree"] >= min_out_degree)
    condition_cluster  = df["clustering_coeff"] < max_clustering
    condition_pagerank = df["pagerank"] <= pr_threshold

    # A fake node must have suspicious follow-ratio AND (low community clustering OR low authority)
    df["rule_label"] = (
        condition_ratio & (condition_cluster | condition_pagerank)
    ).astype(int)

    # Build human-readable reasons per node
    def build_reason(row):
        reasons = []
        if row["out_in_ratio"] > out_in_ratio_threshold and row["out_degree"] >= min_out_degree:
            reasons.append(
                f"follows {int(row['out_degree'])} users but only {int(row['in_degree'])} follow back "
                f"(ratio {row['out_in_ratio']:.1f}x)"
            )
        if row["clustering_coeff"] < max_clustering:
            reasons.append(
                f"very low community clustering ({row['clustering_coeff']:.4f}), "
                "suggesting isolated/spammy behavior"
            )
        if row["pagerank"] < pr_threshold:
            reasons.append(
                f"low PageRank ({row['pagerank']:.6f}), indicating low network authority"
            )
        if not reasons:
            return "No suspicious signals detected."
        return "This account is flagged because: " + "; ".join(reasons) + "."

    df["rule_reasons"] = df.apply(build_reason, axis=1)

    flagged = df["rule_label"].sum()
    logger.info(f"Rule-based detection: {flagged} / {len(df)} nodes flagged as fake ({100*flagged/len(df):.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# B. ML-BASED DETECTION (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════

def ml_detection(
    df: pd.DataFrame,
    test_size: float = 0.3,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Train a RandomForest classifier using 'rule_label' as pseudo-labels,
    then predict on the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with 'rule_label' column (from rule_based_detection).
    test_size : float
        Fraction of data reserved for evaluation.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    (df_with_predictions, metrics_dict)
    """
    df = df.copy()

    X = df[FEATURE_COLS].fillna(0)
    y = df["rule_label"]

    # Need at least 2 classes
    if y.nunique() < 2:
        logger.warning("Only one class present — skipping ML training, using rule labels.")
        df["ml_label"] = y
        df["ml_probability"] = y.astype(float)
        df["ml_reasons"] = df["rule_reasons"]
        return df, {"note": "Only one class found. ML skipped."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scale + forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",   # handles imbalanced fake/real ratio
            n_jobs=-1,                 # use all CPU cores
            random_state=random_state,
        )),
    ])

    logger.info(f"Training RandomForest on {len(X_train):,} samples...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate on test split ───────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }
    logger.info(
        f"ML Metrics → Accuracy: {metrics['accuracy']}, "
        f"Precision: {metrics['precision']}, Recall: {metrics['recall']}, "
        f"F1: {metrics['f1_score']}"
    )

    # ── Predict on full dataset ──────────────────────────────────────────
    df["ml_label"]       = pipeline.predict(X)
    df["ml_probability"] = pipeline.predict_proba(X)[:, 1].round(4)

    # ── Feature importances for explainability ───────────────────────────
    rf_model = pipeline.named_steps["clf"]
    importances = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
    top_features = sorted(importances, key=importances.get, reverse=True)

    def build_ml_reason(row):
        if row["ml_label"] == 0:
            return "Account appears legitimate based on graph features."
        reasons = []
        for feat in top_features[:3]:   # top-3 most important features
            val = row[feat]
            if feat == "out_in_ratio" and val > 5:
                reasons.append(f"high follow-to-follower ratio ({val:.1f}x)")
            elif feat == "clustering_coeff" and val < 0.1:
                reasons.append(f"low clustering ({val:.4f})")
            elif feat == "pagerank" and val < 1e-4:
                reasons.append(f"low PageRank ({val:.6f})")
            elif feat == "in_degree" and val < 10:
                reasons.append(f"very few followers ({int(val)})")
            elif feat == "out_degree" and val > 100:
                reasons.append(f"follows many accounts ({int(val)})")
            elif feat == "betweenness" and val < 1e-5:
                reasons.append(f"near-zero betweenness ({val:.8f})")
        if not reasons:
            reasons = ["combination of suspicious graph metrics"]
        return "This account is flagged because: " + ", ".join(reasons) + "."

    df["ml_reasons"] = df.apply(build_ml_reason, axis=1)

    return df, metrics


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED LABEL: fake if EITHER method flags it
# ═══════════════════════════════════════════════════════════════════════════

def combine_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a final 'is_fake' label: 1 if rule OR ML flags the account.
    Also selects the best reason string.
    """
    df = df.copy()
    df["is_fake"] = ((df["rule_label"] == 1) | (df["ml_label"] == 1)).astype(int)

    def best_reason(row):
        if row["ml_label"] == 1:
            return row["ml_reasons"]
        if row["rule_label"] == 1:
            return row["rule_reasons"]
        return "Account appears legitimate."

    df["reason"] = df.apply(best_reason, axis=1)
    return df
