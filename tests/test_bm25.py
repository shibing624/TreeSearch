# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.rank_bm25 module.

Tests cover: tokenize, BM25Okapi, NodeBM25Index (hierarchical weighting + ancestor propagation).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from treesearch.rank_bm25 import tokenize, BM25Okapi, NodeBM25Index
from treesearch.tree import Document


class TestTokenize:
    def test_english_basic(self):
        tokens = tokenize("Hello world, this is a test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_english_special_chars(self):
        tokens = tokenize("FastAPI + React.js (v18)")
        assert "fastapi" in tokens
        assert "react" in tokens or "React" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_chinese_text(self):
        tokens = tokenize("这是一个测试文档")
        assert len(tokens) > 0
        # Should have Chinese tokens (either jieba segments or individual chars)
        assert any(len(t) >= 1 for t in tokens)

    def test_mixed_chinese_english(self):
        tokens = tokenize("使用FastAPI构建后端服务")
        assert len(tokens) > 0

    def test_whitespace_only(self):
        tokens = tokenize("   ")
        assert tokens == []


class TestBM25Okapi:
    def test_basic_scoring(self):
        corpus = [
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "dog"],
            ["quick", "brown", "fox", "jumps"],
            ["the", "cat", "sat", "mat"],
            ["fox", "hunt", "forest"],
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["quick", "fox"])
        # Doc 0 and 2 mention both words, doc 1 mentions neither
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_top_n(self):
        corpus = [
            ["python", "backend"],
            ["react", "frontend"],
            ["python", "machine", "learning"],
        ]
        bm25 = BM25Okapi(corpus)
        top = bm25.get_top_n(["python"], n=2)
        assert len(top) == 2
        # Top results should be doc 0 and 2 (both mention python)
        top_indices = {idx for idx, _ in top}
        assert 0 in top_indices
        assert 2 in top_indices

    def test_empty_corpus(self):
        bm25 = BM25Okapi([])
        assert bm25.corpus_size == 0

    def test_single_doc(self):
        bm25 = BM25Okapi([["hello", "world"]])
        scores = bm25.get_scores(["hello"])
        assert len(scores) == 1
        # Single doc corpus: score should be non-negative after epsilon correction
        assert isinstance(scores[0], float)

    def test_query_not_in_corpus(self):
        corpus = [["apple", "banana"], ["cherry", "date"]]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(["xyz"])
        assert all(s == 0 for s in scores)


class TestNodeBM25Index:
    @pytest.fixture
    def sample_documents(self, sample_tree_structure):
        doc = Document(
            doc_id="test_doc",
            doc_name="Test Doc",
            structure=sample_tree_structure,
        )
        return [doc]

    def test_search_returns_results(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        results = index.search("backend python FastAPI")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "node_id" in r
            assert "doc_id" in r
            assert "bm25_score" in r

    def test_search_relevance_order(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        results = index.search("backend python")
        if len(results) >= 2:
            scores = [r["bm25_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        results = index.search("")
        assert results == []

    def test_search_no_match(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        results = index.search("xyznonexistent12345")
        assert results == []

    def test_top_k_limit(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        results = index.search("system", top_k=2)
        assert len(results) <= 2

    def test_get_node_scores_for_doc(self, sample_documents):
        index = NodeBM25Index(sample_documents)
        scores = index.get_node_scores_for_doc("backend", "test_doc")
        assert isinstance(scores, dict)
        for nid, score in scores.items():
            assert isinstance(nid, str)
            assert isinstance(score, float)

    def test_ancestor_propagation(self, sample_tree_structure):
        """Parent node should get score boost from children."""
        doc = Document(doc_id="test", doc_name="Test", structure=sample_tree_structure)
        index_with = NodeBM25Index([doc], ancestor_decay=0.5)
        index_without = NodeBM25Index([doc], ancestor_decay=0.0)

        results_with = index_with.search("backend python", propagate=True)
        results_without = index_without.search("backend python", propagate=False)

        # Parent "Architecture" (0000) should have higher score with propagation
        score_with = {r["node_id"]: r["bm25_score"] for r in results_with}
        score_without = {r["node_id"]: r["bm25_score"] for r in results_without}

        if "0000" in score_with and "0000" in score_without:
            assert score_with["0000"] >= score_without["0000"]

    def test_hierarchical_weighting(self, sample_tree_structure):
        """Title matches should score higher than body matches."""
        doc = Document(doc_id="test", doc_name="Test", structure=sample_tree_structure)
        index = NodeBM25Index([doc], title_weight=1.0, summary_weight=0.7, body_weight=0.3)
        results = index.search("Backend")
        # "Backend" node (0001) has it in title, should rank high
        if results:
            assert results[0]["node_id"] == "0001"

    def test_chinese_query(self, sample_tree_structure):
        """Chinese queries should work without errors."""
        doc = Document(doc_id="test", doc_name="Test", structure=sample_tree_structure)
        index = NodeBM25Index([doc])
        results = index.search("后端技术架构")
        assert isinstance(results, list)

    def test_multi_doc_index(self, sample_tree_structure):
        """Index across multiple documents."""
        doc_a = Document(doc_id="a", doc_name="Doc A", structure=sample_tree_structure)
        doc_b = Document(
            doc_id="b", doc_name="Doc B",
            structure=[{
                "title": "Machine Learning",
                "summary": "ML algorithms and models.",
                "node_id": "0000",
                "text": "Deep learning neural networks training.",
            }],
        )
        index = NodeBM25Index([doc_a, doc_b])

        # Query about ML should rank doc_b higher
        results = index.search("machine learning neural")
        if results:
            assert results[0]["doc_id"] == "b"

        # Query about backend should rank doc_a higher
        results = index.search("backend python FastAPI")
        if results:
            assert results[0]["doc_id"] == "a"
