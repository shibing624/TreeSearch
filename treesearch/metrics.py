# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Retrieval evaluation metrics for benchmark comparison.

Supports: Precision@K, Recall@K, MRR, NDCG@K, Hit@K.
"""
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Precision@K: fraction of top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / len(relevant_set)


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> bool:
    """Hit@K: whether at least one relevant item is in top-k."""
    top_k = set(retrieved[:k])
    return any(r in top_k for r in relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """Reciprocal Rank: 1/rank of first relevant item (0 if none found)."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain at K."""
    if not relevant or k <= 0:
        return 0.0
    relevant_set = set(relevant)

    # DCG
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if r in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_query(
    retrieved_node_ids: list[str],
    relevant_node_ids: list[str],
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Evaluate a single query's retrieval results.

    Args:
        retrieved_node_ids: ordered list of retrieved node IDs
        relevant_node_ids: ground truth relevant node IDs
        k_values: list of K values for @K metrics (default: [1, 3, 5])

    Returns:
        dict with all metrics
    """
    if k_values is None:
        k_values = [1, 3, 5]

    result = {
        "mrr": reciprocal_rank(retrieved_node_ids, relevant_node_ids),
    }

    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(retrieved_node_ids, relevant_node_ids, k)
        result[f"recall@{k}"] = recall_at_k(retrieved_node_ids, relevant_node_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_node_ids, relevant_node_ids, k)
        result[f"hit@{k}"] = 1.0 if hit_at_k(retrieved_node_ids, relevant_node_ids, k) else 0.0
        result[f"f1@{k}"] = f1_at_k(retrieved_node_ids, relevant_node_ids, k)

    return result


def evaluate_benchmark(
    query_results: list[dict],
    k_values: Optional[list[int]] = None,
) -> dict:
    """
    Evaluate a full benchmark (multiple queries).

    Args:
        query_results: list of dicts, each with:
            - retrieved: list of retrieved node_ids
            - relevant: list of ground truth node_ids
        k_values: list of K values

    Returns:
        dict with averaged metrics across all queries
    """
    if k_values is None:
        k_values = [1, 3, 5]

    if not query_results:
        return {}

    all_metrics = []
    for qr in query_results:
        m = evaluate_query(qr["retrieved"], qr["relevant"], k_values)
        all_metrics.append(m)

    # Average across queries
    avg = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg
