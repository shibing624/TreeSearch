# -*- coding: utf-8 -*-
"""Metrics aggregation for GraphRAG paper experiments."""

from treesearch.rag import RealRepoEvalResult, aggregate_eval_results


def aggregate_method_summary(
    results: list[RealRepoEvalResult],
    latency_seconds: float,
    llm_calls: int,
) -> dict[str, float | int]:
    summary = aggregate_eval_results(results)
    count = int(summary["count"])
    summary.update(
        {
            "latency_seconds": latency_seconds,
            "avg_latency_seconds": latency_seconds / count if count else 0.0,
            "llm_calls": llm_calls,
            "llm_calls_per_query": llm_calls / count if count else 0.0,
        }
    )
    return summary
