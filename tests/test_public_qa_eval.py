# -*- coding: utf-8 -*-
"""Tests for public multi-hop QA evaluation harness."""

from pathlib import Path

from evaluation.evaluate import evaluate_public_qa, load_dataset


def test_load_dataset_reads_questions_and_corpus():
    questions, corpus = load_dataset("test_sample", Path("evaluation/data"))

    assert len(questions) == 10
    assert len(corpus) == 20
    assert questions[0]["question"]
    assert corpus[0]["title"]


def test_evaluate_public_qa_returns_method_summaries(tmp_path):
    report = evaluate_public_qa(
        dataset_name="test_sample",
        data_dir=Path("evaluation/data"),
        methods=("treesearch", "fts5", "dense", "hybrid"),
        max_samples=5,
        output_path=tmp_path / "results.json",
        markdown_path=tmp_path / "results.md",
    )

    assert set(report["summary"]) == {"treesearch", "fts5", "dense", "hybrid"}
    assert report["summary"]["treesearch"]["count"] == 5
    assert "recall@5" in report["summary"]["treesearch"]
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()
