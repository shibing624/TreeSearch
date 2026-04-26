# -*- coding: utf-8 -*-
"""Tests for GraphRAG experiment runner outputs."""

from pathlib import Path

from examples.graphrag.real_repo_bench import main, run


def test_real_repo_bench_runs_graph_and_treesearch_baselines(tmp_path):
    output_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"

    report = run(
        paths=["examples/graphrag/fixtures/repo"],
        queries_path=Path("examples/graphrag/fixtures/queries.json"),
        triplets_path=Path("examples/graphrag/fixtures/triplets.json"),
        output_path=output_path,
        markdown_path=markdown_path,
        baseline="both",
        graph_store="sqlite",
        graph_store_path=tmp_path / "graph.db",
        use_llm=False,
    )

    assert output_path.exists()
    assert markdown_path.exists()
    assert "graphrag" in report["summary"]
    assert "treesearch" in report["summary"]
    assert report["summary"]["graphrag"]["count"] == 30
    assert report["summary"]["treesearch"]["count"] == 30
    assert report["summary"]["graphrag"]["latency_seconds"] >= 0.0
    assert report["summary"]["graphrag"]["avg_latency_seconds"] >= 0.0
    assert report["summary"]["graphrag"]["llm_calls"] == 0
    assert "| method | count | node_recall |" in markdown_path.read_text(encoding="utf-8")
    assert "latency_seconds" in markdown_path.read_text(encoding="utf-8")


def test_real_repo_bench_defaults_work_outside_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["real_repo_bench.py"])

    main()

    assert (tmp_path / "output" / "graphrag_real_repo_results.json").exists()
    assert (tmp_path / "output" / "graphrag_real_repo_results.md").exists()
