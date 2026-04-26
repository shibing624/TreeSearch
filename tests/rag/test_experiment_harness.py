# -*- coding: utf-8 -*-
"""Tests for paper-grade GraphRAG experiment harness helpers."""

from pathlib import Path

from examples.graphrag.baselines import baseline_methods
from examples.graphrag.metrics import aggregate_method_summary
from examples.graphrag.real_repo_bench import run
from examples.graphrag.report import format_latex_summary, format_markdown_summary
from treesearch.rag.eval import RealRepoEvalResult


def test_baseline_registry_includes_paper_minimum_methods():
    names = baseline_methods()

    assert "graphrag" in names
    assert "treesearch" in names
    assert "fts5" in names
    assert "dense" in names
    assert "hybrid" in names


def test_metric_summary_includes_runtime_and_efficiency_fields():
    result = RealRepoEvalResult(
        query_id="q1",
        node_recall=1.0,
        source_path_recall=1.0,
        citation_precision=0.5,
        citation_recall=1.0,
        line_grounding_accuracy=1.0,
        verification_ok=True,
        task_success=True,
    )

    summary = aggregate_method_summary([result], latency_seconds=2.0, llm_calls=3)

    assert summary["count"] == 1
    assert summary["avg_latency_seconds"] == 2.0
    assert summary["llm_calls_per_query"] == 3.0


def test_report_formats_markdown_and_latex_tables():
    report = {
        "summary": {
            "ours": {
                "count": 1,
                "node_recall": 1.0,
                "source_path_recall": 1.0,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "line_grounding_accuracy": 1.0,
                "task_success_rate": 1.0,
                "latency_seconds": 0.1,
                "avg_latency_seconds": 0.1,
                "llm_calls": 2,
                "llm_calls_per_query": 2.0,
            }
        }
    }

    assert "| method | count |" in format_markdown_summary(report)
    assert "\\begin{tabular}" in format_latex_summary(report)


def test_real_repo_bench_all_methods_on_mini_fixture(tmp_path):
    report = run(
        paths=["examples/graphrag/fixtures/repo"],
        queries_path=Path("examples/graphrag/fixtures/queries.json"),
        triplets_path=Path("examples/graphrag/fixtures/triplets.json"),
        output_path=tmp_path / "results.json",
        markdown_path=tmp_path / "results.md",
        latex_path=tmp_path / "results.tex",
        baseline="all",
        graph_store="sqlite",
        graph_store_path=tmp_path / "graph.db",
        use_llm=False,
    )

    assert report["summary"]["graphrag"]["count"] >= 30
    assert set(report["summary"]) >= {"graphrag", "treesearch", "fts5", "dense", "hybrid"}
    assert (tmp_path / "results.tex").exists()
