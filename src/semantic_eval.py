"""
semantic_eval.py
----------------
Semantic evaluation of emoji predictions using EmoSim508 similarity clusters.

Paper reference (Section 3.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rather than treating each emoji as an independent class, we group emojis into
semantic clusters using pre-computed pairwise similarity scores from EmoSim508
(derived from 147 million tweets).  Two emojis belong to the same cluster when
their similarity score is ≥ 0.7.  Prediction performance is then measured at
the *cluster* level instead of the *emoji* level.

EmoSim508 format expected
~~~~~~~~~~~~~~~~~~~~~~~~~
The file ``emoji_similarity.json`` should be a list of
    [emoji_1, emoji_2, similarity_score]
triples, e.g.:
    [["😀", "😃", 0.92], ["😀", "😄", 0.88], ...]
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.evaluation import metrics_at_k
from src.model import get_topk_predictions


# Cluster construction

def build_emoji_clusters(
    similarity_triples: list,
    threshold: float = 0.7,
) -> dict:
    """
    Build a mapping from each emoji to a canonical cluster ID using
    single-linkage clustering on EmoSim508 similarity scores.

    Parameters
    ----------
    similarity_triples : list of [emoji_a, emoji_b, score]
    threshold          : minimum similarity to place two emojis in the same cluster

    Returns
    -------
    dict  {emoji_str: cluster_id (int)}
    """
    # Union-Find (path-compressed)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for e1, e2, score in similarity_triples:
        if float(score) >= threshold:
            union(str(e1), str(e2))

    # Assign integer cluster IDs
    root_to_id: dict = {}
    cluster_map: dict = {}
    for emoji in list(parent.keys()):
        root = find(emoji)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        cluster_map[emoji] = root_to_id[root]

    return cluster_map


# Cluster-level evaluation

def evaluate_semantic(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    label_encoder,
    cluster_map: dict,
    num_classes: int = 20,
) -> dict:
    """
    Evaluate predictions at the semantic-cluster level.

    Emojis not present in *cluster_map* are assigned their own singleton
    cluster (no similarity relationships).

    Returns
    -------
    dict with keys matching the standard evaluate() output, but measured
    over cluster IDs rather than individual emoji IDs.
    """
    actual, top1, top3 = get_topk_predictions(
        X_test, y_test, checkpoint_path, label_encoder, k=3, num_classes=num_classes
    )
    _, _, top5 = get_topk_predictions(
        X_test, y_test, checkpoint_path, label_encoder, k=5, num_classes=num_classes
    )

    # Build a fallback cluster map that assigns singletons for unknown emojis
    all_emojis = set(actual) | {e for preds in top3 for e in preds}
    next_id = max(cluster_map.values(), default=-1) + 1
    full_map: dict = dict(cluster_map)
    for e in all_emojis:
        if e not in full_map:
            full_map[e] = next_id
            next_id += 1

    # Convert to cluster IDs
    y_true_c  = np.array([full_map[e] for e in actual])
    y_top1_c  = np.array([full_map[e] for e in top1])
    y_top3_c  = np.array([[full_map[e] for e in preds] for preds in top3])
    y_top5_c  = np.array([[full_map[e] for e in preds] for preds in top5])

    # Build a fake probability matrix for metrics_at_k
    num_clusters = next_id
    probs_top3 = _topk_to_prob_matrix(y_top3_c, num_clusters, len(actual))
    probs_top5 = _topk_to_prob_matrix(y_top5_c, num_clusters, len(actual))

    acc  = float(accuracy_score(y_true_c, y_top1_c))
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true_c, y_top1_c, average="weighted", zero_division=0
    )
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true_c, y_top1_c, average="macro", zero_division=0
    )

    top3_m = metrics_at_k(y_true_c, probs_top3, k=3)
    top5_m = metrics_at_k(y_true_c, probs_top5, k=5)

    return {
        "Cluster Accuracy":           acc,
        "Weighted Precision":         float(p_w),
        "Weighted Recall":            float(r_w),
        "Weighted F1":                float(f_w),
        "Macro Precision":            float(p_ma),
        "Macro F1":                   float(f_ma),
        "Top-3 Cluster Accuracy":     top3_m["Accuracy@K"],
        "Top-3 Weighted Precision":   top3_m["Weighted Precision@K"],
        "Top-3 Weighted F1":          top3_m["Weighted F1@K"],
        "Top-5 Cluster Accuracy":     top5_m["Accuracy@K"],
        "Top-5 Weighted Precision":   top5_m["Weighted Precision@K"],
        "Top-5 Weighted F1":          top5_m["Weighted F1@K"],
    }


def _topk_to_prob_matrix(topk_indices: np.ndarray, num_classes: int, n: int) -> np.ndarray:
    """
    Convert a (n, k) array of top-k class indices into a (n, num_classes)
    soft probability matrix suitable for metrics_at_k.
    Each predicted class gets equal weight 1/k.
    """
    k = topk_indices.shape[1]
    probs = np.zeros((n, num_classes), dtype=np.float32)
    for i in range(n):
        for j, cls in enumerate(topk_indices[i]):
            if cls < num_classes:
                probs[i, cls] += 1.0 / k
    return probs


# Raw similarity-based scoring (original approach)

def avg_emoji_similarity(
    predicted_list: list,
    actual_list: list,
    similarity_triples: list,
) -> float:
    """
    Compute the average best-match similarity between predicted and actual emojis.

    For each sample, find the highest similarity score among all predicted
    emojis and the actual emoji, then average across samples.

    Parameters
    ----------
    predicted_list   : list of str or list-of-str (top-k predictions per sample)
    actual_list      : list of str (one actual emoji per sample)
    similarity_triples : list of [e1, e2, score] from EmoSim508

    Returns
    -------
    float  – mean best-match similarity (0 if no pair found)
    """
    sim_lookup: dict = {}
    for e1, e2, score in similarity_triples:
        sim_lookup[(str(e1), str(e2))] = float(score)
        sim_lookup[(str(e2), str(e1))] = float(score)

    total, count = 0.0, 0
    for preds, actual in zip(predicted_list, actual_list):
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        if not isinstance(preds, list):
            preds = [preds]
        actual_str = str(actual.item() if hasattr(actual, "item") else actual)

        best = max(
            (sim_lookup.get((str(p), actual_str), 0.0) for p in preds),
            default=0.0,
        )
        if best > 0:
            total += best
            count += 1

    return total / count if count > 0 else 0.0


def get_similarity_scores(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    label_encoder,
    similarity_triples: list,
    num_classes: int = 20,
) -> dict:
    """
    Compute top-1, top-3, and top-5 average similarity scores for a model.
    """
    actual, top1, top3 = get_topk_predictions(
        X_test, y_test, checkpoint_path, label_encoder, k=3, num_classes=num_classes
    )
    _, _, top5 = get_topk_predictions(
        X_test, y_test, checkpoint_path, label_encoder, k=5, num_classes=num_classes
    )
    return {
        "top1_avg_similarity": avg_emoji_similarity(top1,  actual, similarity_triples),
        "top3_avg_similarity": avg_emoji_similarity(top3,  actual, similarity_triples),
        "top5_avg_similarity": avg_emoji_similarity(top5,  actual, similarity_triples),
    }
