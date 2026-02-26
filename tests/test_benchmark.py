# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.benchmark module.
"""
import json
import os
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from treesearch.benchmark import (
    BenchmarkSample,
    BenchmarkReport,
    SampleResult,
    load_qasper,
    load_quality,
    load_custom,
    load_dataset,
    resolve_relevant_nodes,
    print_report,
    print_comparison,
)
from treesearch.metrics import CostStats, CostTracker, aggregate_cost_stats
from treesearch.tree import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    return [Document(
        doc_id="paper_001",
        doc_name="Test Paper",
        structure=[
            {
                "title": "Introduction",
                "summary": "Paper introduction.",
                "node_id": "0000",
                "text": "This paper presents a novel approach to information retrieval.",
                "nodes": [
                    {
                        "title": "Background",
                        "summary": "Background information.",
                        "node_id": "0001",
                        "text": "Information retrieval is a well-studied field.",
                    },
                ],
            },
            {
                "title": "Methods",
                "summary": "Research methods.",
                "node_id": "0002",
                "text": "We propose tree-based search over document structures.",
                "nodes": [
                    {
                        "title": "Tree Construction",
                        "summary": "How trees are built.",
                        "node_id": "0003",
                        "text": "Documents are parsed into hierarchical trees.",
                    },
                ],
            },
            {
                "title": "Results",
                "summary": "Experimental results.",
                "node_id": "0004",
                "text": "Our method outperforms baselines on QASPER and QuALITY.",
            },
        ],
    )]


@pytest.fixture
def qasper_data_file():
    """Create a temporary QASPER-format JSON file."""
    data = {
        "paper_001": {
            "title": "Test Paper",
            "abstract": "A test paper.",
            "full_text": {
                "section_name": ["Introduction", "Methods", "Results"],
                "paragraphs": [["Paragraph 1"], ["Paragraph 2"], ["Paragraph 3"]],
            },
            "qas": [
                {
                    "question": "What method is proposed?",
                    "question_type": "extractive",
                    "answers": [
                        {
                            "answer": {
                                "free_form_answer": "tree-based search",
                                "extractive_spans": ["tree-based search"],
                            },
                            "evidence": ["Methods"],
                        }
                    ],
                },
                {
                    "question": "What datasets are used?",
                    "question_type": "abstractive",
                    "answers": [
                        {
                            "answer": {
                                "free_form_answer": "QASPER and QuALITY",
                                "extractive_spans": [],
                            },
                            "evidence": ["Results"],
                        }
                    ],
                },
            ],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def quality_data_file():
    """Create a temporary QuALITY-format JSONL file."""
    items = [
        {
            "article_id": "art_001",
            "article": "A long article about machine learning...",
            "questions": [
                {
                    "question": "What is the main topic?",
                    "options": ["Physics", "Machine Learning", "Biology", "Chemistry"],
                    "gold_label": 2,
                    "difficult": 0,
                },
                {
                    "question": "Which technique is used?",
                    "options": ["CNN", "RNN", "Transformer", "Decision Tree"],
                    "gold_label": 3,
                    "difficult": 1,
                },
            ],
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def custom_data_file():
    """Create a temporary custom JSONL file."""
    items = [
        {
            "question": "What is TreeSearch?",
            "answer": "A tree-based search method.",
            "relevant_section_titles": ["Methods"],
            "doc_id": "paper_001",
            "question_type": "factual",
        },
        {
            "question": "How does indexing work?",
            "answer": "Documents are parsed into trees.",
            "evidence_texts": ["hierarchical trees"],
            "doc_id": "paper_001",
            "question_type": "factual",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: Data structures
# ---------------------------------------------------------------------------

class TestDataStructures:
    def test_benchmark_sample_defaults(self):
        s = BenchmarkSample()
        assert s.question == ""
        assert s.evidence_texts == []

    def test_benchmark_sample_with_data(self):
        s = BenchmarkSample(
            question="What?", answer="That.",
            evidence_texts=["sect1"], doc_id="doc1",
        )
        assert s.question == "What?"
        assert s.doc_id == "doc1"

    def test_sample_result_defaults(self):
        r = SampleResult()
        assert r.retrieved_node_ids == []
        assert r.cost.total_tokens == 0

    def test_benchmark_report_to_dict(self):
        report = BenchmarkReport(
            dataset="qasper", strategy="bm25", model="gpt-4o-mini",
            num_samples=10,
            avg_retrieval_metrics={"mrr": 0.8},
            avg_cost=CostStats(total_tokens=100, llm_calls=5),
        )
        d = report.to_dict()
        assert d["dataset"] == "qasper"
        assert d["strategy"] == "bm25"
        assert d["avg_cost"]["total_tokens"] == 100


# ---------------------------------------------------------------------------
# Tests: Cost tracking
# ---------------------------------------------------------------------------

class TestCostTracking:
    def test_cost_stats_defaults(self):
        cs = CostStats()
        assert cs.total_tokens == 0
        assert cs.latency_seconds == 0.0

    def test_cost_stats_to_dict(self):
        cs = CostStats(total_tokens=100, llm_calls=3, latency_seconds=1.234)
        d = cs.to_dict()
        assert d["total_tokens"] == 100
        assert d["llm_calls"] == 3
        assert d["latency_seconds"] == 1.234

    def test_cost_tracker_context_manager(self):
        tracker = CostTracker()
        with tracker:
            pass  # simulate work
        assert tracker.stats.latency_seconds >= 0.0

    def test_cost_tracker_record_llm_call(self):
        tracker = CostTracker()
        tracker.record_llm_call(prompt_tokens=50, completion_tokens=30)
        tracker.record_llm_call(prompt_tokens=40, completion_tokens=20)
        assert tracker.stats.total_tokens == 140
        assert tracker.stats.prompt_tokens == 90
        assert tracker.stats.completion_tokens == 50
        assert tracker.stats.llm_calls == 2

    def test_aggregate_cost_stats(self):
        stats = [
            CostStats(total_tokens=100, llm_calls=2, latency_seconds=1.0),
            CostStats(total_tokens=200, llm_calls=4, latency_seconds=3.0),
        ]
        avg = aggregate_cost_stats(stats)
        assert avg.total_tokens == 150
        assert avg.llm_calls == 3
        assert avg.latency_seconds == pytest.approx(2.0)

    def test_aggregate_cost_stats_empty(self):
        avg = aggregate_cost_stats([])
        assert avg.total_tokens == 0


# ---------------------------------------------------------------------------
# Tests: Dataset loaders
# ---------------------------------------------------------------------------

class TestLoadQasper:
    def test_load_basic(self, qasper_data_file):
        samples = load_qasper(qasper_data_file)
        assert len(samples) == 2
        assert samples[0].question == "What method is proposed?"
        assert samples[0].question_type == "extractive"
        assert "tree-based search" in samples[0].answer
        assert samples[0].doc_id == "paper_001"

    def test_load_max_samples(self, qasper_data_file):
        samples = load_qasper(qasper_data_file, max_samples=1)
        assert len(samples) == 1


class TestLoadQuality:
    def test_load_basic(self, quality_data_file):
        samples = load_quality(quality_data_file)
        assert len(samples) == 2
        assert samples[0].question == "What is the main topic?"
        assert samples[0].answer == "Machine Learning"
        assert samples[0].question_type == "multiple_choice"
        assert samples[1].answer == "Transformer"

    def test_load_max_samples(self, quality_data_file):
        samples = load_quality(quality_data_file, max_samples=1)
        assert len(samples) == 1


class TestLoadCustom:
    def test_load_basic(self, custom_data_file):
        samples = load_custom(custom_data_file)
        assert len(samples) == 2
        assert samples[0].question == "What is TreeSearch?"
        assert samples[0].relevant_section_titles == ["Methods"]

    def test_load_max_samples(self, custom_data_file):
        samples = load_custom(custom_data_file, max_samples=1)
        assert len(samples) == 1


class TestLoadDataset:
    def test_qasper(self, qasper_data_file):
        samples = load_dataset("qasper", qasper_data_file)
        assert len(samples) == 2

    def test_quality(self, quality_data_file):
        samples = load_dataset("quality", quality_data_file)
        assert len(samples) == 2

    def test_custom(self, custom_data_file):
        samples = load_dataset("custom", custom_data_file)
        assert len(samples) == 2

    def test_unknown_dataset(self, custom_data_file):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("hotpot_qa", custom_data_file)


# ---------------------------------------------------------------------------
# Tests: Ground truth resolution
# ---------------------------------------------------------------------------

class TestResolveRelevantNodes:
    def test_resolve_by_title(self, sample_documents):
        sample = BenchmarkSample(
            question="test",
            relevant_section_titles=["Methods"],
            doc_id="paper_001",
        )
        ids = resolve_relevant_nodes(sample, sample_documents)
        assert "0002" in ids

    def test_resolve_by_evidence_text(self, sample_documents):
        sample = BenchmarkSample(
            question="test",
            evidence_texts=["hierarchical trees"],
            doc_id="paper_001",
        )
        ids = resolve_relevant_nodes(sample, sample_documents)
        assert "0003" in ids

    def test_resolve_no_match(self, sample_documents):
        sample = BenchmarkSample(
            question="test",
            relevant_section_titles=["Nonexistent Section"],
        )
        ids = resolve_relevant_nodes(sample, sample_documents)
        assert ids == []

    def test_resolve_doc_id_filter(self, sample_documents):
        sample = BenchmarkSample(
            question="test",
            relevant_section_titles=["Methods"],
            doc_id="wrong_paper",
        )
        # doc_id doesn't match, falls back to all documents
        ids = resolve_relevant_nodes(sample, sample_documents)
        assert "0002" in ids


# ---------------------------------------------------------------------------
# Tests: Report printing (smoke tests)
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_print_report(self, capsys):
        report = BenchmarkReport(
            dataset="qasper", strategy="bm25", model="gpt-4o-mini",
            num_samples=5,
            avg_retrieval_metrics={"mrr": 0.75, "precision@3": 0.6},
            avg_cost=CostStats(llm_calls=0, total_tokens=0, latency_seconds=0.5),
            total_cost=CostStats(llm_calls=0, total_tokens=0, latency_seconds=2.5),
        )
        print_report(report)
        captured = capsys.readouterr()
        assert "qasper" in captured.out
        assert "BM25" in captured.out
        assert "0.7500" in captured.out

    def test_print_comparison(self, capsys):
        reports = [
            BenchmarkReport(
                dataset="qasper", strategy="bm25", model="gpt-4o-mini",
                num_samples=5,
                avg_retrieval_metrics={"mrr": 0.7, "precision@3": 0.5, "recall@3": 0.6,
                                       "ndcg@3": 0.65, "hit@1": 0.4, "f1@3": 0.55},
                avg_cost=CostStats(llm_calls=0, latency_seconds=0.1),
                total_cost=CostStats(llm_calls=0, latency_seconds=0.5),
            ),
            BenchmarkReport(
                dataset="qasper", strategy="best_first", model="gpt-4o-mini",
                num_samples=5,
                avg_retrieval_metrics={"mrr": 0.85, "precision@3": 0.7, "recall@3": 0.8,
                                       "ndcg@3": 0.78, "hit@1": 0.6, "f1@3": 0.74},
                avg_cost=CostStats(llm_calls=10, latency_seconds=2.0),
                total_cost=CostStats(llm_calls=50, latency_seconds=10.0),
            ),
        ]
        print_comparison(reports)
        captured = capsys.readouterr()
        assert "BENCHMARK COMPARISON" in captured.out
        assert "BM25" in captured.out
        assert "BEST_FIRST" in captured.out

    def test_print_comparison_empty(self, capsys):
        print_comparison([])
        captured = capsys.readouterr()
        assert captured.out == ""
