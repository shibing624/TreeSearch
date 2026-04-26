# -*- coding: utf-8 -*-
"""Tests for public multi-hop QA evaluation harness."""

from pathlib import Path
import urllib.error

import pytest

from evaluation.evaluate import EmbeddingCache, answer_metrics, embed_with_cache, evaluate_public_qa, load_dataset


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append([
                float("einstein" in normalized or "relativity" in normalized),
                float("paris" in normalized or "france" in normalized),
            ])
        return vectors


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
        methods=("treesearch", "fts5", "dense", "hybrid", "graphrag"),
        max_samples=5,
        output_path=tmp_path / "results.json",
        markdown_path=tmp_path / "results.md",
        embedding_client=FakeEmbeddingClient(),
    )

    assert set(report["summary"]) == {"treesearch", "fts5", "dense", "hybrid", "graphrag"}
    assert report["summary"]["treesearch"]["count"] == 5
    assert "recall@5" in report["summary"]["treesearch"]
    assert "answer_accuracy" in report["summary"]["graphrag"]
    assert "answer_f1" in report["summary"]["graphrag"]
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()


def test_public_qa_reports_graphrag_ablation_methods(tmp_path):
    report = evaluate_public_qa(
        dataset_name="test_sample",
        data_dir=Path("evaluation/data"),
        methods=("graphrag", "graphrag_no_structure", "graphrag_no_entity"),
        max_samples=3,
        output_path=tmp_path / "results.json",
        markdown_path=tmp_path / "results.md",
    )

    assert set(report["summary"]) == {"graphrag", "graphrag_no_structure", "graphrag_no_entity"}
    assert report["ablation_groups"]["public_graphrag"] == [
        "graphrag",
        "graphrag_no_structure",
        "graphrag_no_entity",
    ]
    assert report["summary"]["graphrag_no_structure"]["count"] == 3


def test_answer_metrics_include_exact_match_accuracy_and_token_f1():
    metrics = answer_metrics(
        gold_answer="1905",
        predicted_answer="He published the special theory of relativity in 1905.",
    )

    assert metrics["answer_exact_match"] == 0.0
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["answer_f1"] > 0.0


def test_dense_retrieval_uses_embedding_client(tmp_path):
    embedding_client = FakeEmbeddingClient()

    report = evaluate_public_qa(
        dataset_name="test_sample",
        data_dir=Path("evaluation/data"),
        methods=("dense",),
        max_samples=2,
        output_path=tmp_path / "results.json",
        markdown_path=tmp_path / "results.md",
        embedding_client=embedding_client,
        embedding_cache_path=tmp_path / "zhipu_cache.json",
    )

    assert embedding_client.calls
    assert report["summary"]["dense"]["answer_accuracy"] >= 0.0
    assert (tmp_path / "zhipu_cache.json").exists()


def test_embedding_cache_persists_completed_batches_before_api_failure(tmp_path):
    class FlakyEmbeddingClient:
        def __init__(self):
            self.calls = 0

        def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("transient api failure")
            return [[float(len(text))] for text in texts]

    cache_path = tmp_path / "embeddings.json"
    cache = EmbeddingCache(cache_path)

    with pytest.raises(RuntimeError, match="transient api failure"):
        embed_with_cache(["alpha", "beta", "gamma"], FlakyEmbeddingClient(), cache, batch_size=2)

    assert cache_path.exists()
    restored = EmbeddingCache(cache_path)
    assert len(restored.values) == 2


def test_embedding_cache_retries_transient_url_errors(tmp_path):
    class RetryEmbeddingClient:
        def __init__(self):
            self.calls = 0

        def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                raise urllib.error.URLError("transient ssl eof")
            return [[float(len(text))] for text in texts]

    client = RetryEmbeddingClient()
    vectors = embed_with_cache(
        ["alpha", "beta"],
        client,
        EmbeddingCache(tmp_path / "embeddings.json"),
        batch_size=2,
        max_retries=1,
    )

    assert client.calls == 2
    assert vectors == [[5.0], [4.0]]
