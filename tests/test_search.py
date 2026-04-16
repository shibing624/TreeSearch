# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.search module (FTS5-only, no LLM).

Tests cover: search(), search_sync(), GrepFilter, _CombinedScorer.
"""
import pytest
from unittest.mock import patch
from treesearch.search import (
    search,
    search_sync,
    GrepFilter,
    PreFilter,
)
from treesearch.tree import Document


class TestSearchResult:
    def test_fields(self):
        r = {
            "documents": [{"doc_id": "d1", "doc_name": "Doc", "nodes": []}],
            "query": "test",
            "flat_nodes": [],
        }
        assert len(r["documents"]) == 1
        assert r["query"] == "test"
        assert r["flat_nodes"] == []

    def test_empty(self):
        r = {"documents": [], "query": "", "flat_nodes": []}
        assert r["documents"] == []
        assert r["query"] == ""


class TestGrepFilter:
    def test_grep_filter_basic(self, sample_tree_structure):
        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        grep = GrepFilter([doc])
        scores = grep.score_nodes("FastAPI", doc.doc_id)
        # Should find "FastAPI" in the Backend node
        assert len(scores) > 0
        assert any("1" in nid for nid in scores)

    def test_grep_filter_no_match(self, sample_tree_structure):
        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        grep = GrepFilter([doc])
        scores = grep.score_nodes("nonexistent_term_xyz", doc.doc_id)
        assert len(scores) == 0

    def test_grep_filter_case_insensitive(self, sample_tree_structure):
        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        grep = GrepFilter([doc], case_sensitive=False)
        scores = grep.score_nodes("fastapi", doc.doc_id)
        assert len(scores) > 0


class TestSearch:
    @pytest.fixture
    def two_documents(self, sample_tree_structure):
        doc_a = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture and deployment guide.",
            structure=sample_tree_structure,
        )
        doc_b = Document(
            doc_id="b", doc_name="Doc B",
            doc_description="Machine learning algorithms overview.",
            structure=[{
                "title": "ML Basics", "summary": "Introduction to machine learning.",
                "node_id": "0", "text": "Machine learning is a subset of AI.",
            }],
        )
        return [doc_a, doc_b]

    @pytest.mark.asyncio
    async def test_fts5_search_returns_results(self, sample_tree_structure):
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=sample_tree_structure,
        )
        result = await search(
            query="backend Python FastAPI",
            documents=[doc],
        )
        assert isinstance(result, dict)
        assert result["query"] == "backend Python FastAPI"
        assert "documents" in result
        assert "flat_nodes" in result
        # No llm_calls key anymore
        assert "llm_calls" not in result

    @pytest.mark.asyncio
    async def test_fts5_multi_doc(self, two_documents):
        result = await search(
            query="backend architecture",
            documents=two_documents,
            top_k_docs=3,
        )
        assert isinstance(result, dict)
        assert "documents" in result

    @pytest.mark.asyncio
    async def test_flat_nodes_in_result(self, sample_tree_structure):
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=sample_tree_structure,
        )
        result = await search(
            query="backend",
            documents=[doc],
        )
        assert "flat_nodes" in result
        if result["flat_nodes"]:
            node = result["flat_nodes"][0]
            assert "node_id" in node
            assert "doc_id" in node
            assert "doc_name" in node
            assert "title" in node
            assert "score" in node

    @pytest.fixture
    def wildcard_documents(self):
        return [
            Document(
                doc_id="exact",
                doc_name="Exact Auth",
                source_type="text",
                structure=[{
                    "title": "Exact Auth",
                    "summary": "Contains the exact auth token.",
                    "node_id": "0",
                    "text": "Use auth tokens for API access.",
                }],
            ),
            Document(
                doc_id="prefix",
                doc_name="Authentication Guide",
                source_type="text",
                structure=[{
                    "title": "Authentication",
                    "summary": "Authentication and authorizer details.",
                    "node_id": "0",
                    "text": "Authentication depends on an authorizer service.",
                }],
            ),
            Document(
                doc_id="contains",
                doc_name="OAuth Guide",
                source_type="text",
                structure=[{
                    "title": "OAuth",
                    "summary": "OAuth callback handling.",
                    "node_id": "0",
                    "text": "OAuth callbacks must be validated.",
                }],
            ),
        ]

    @pytest.mark.asyncio
    async def test_plain_query_preserves_existing_exact_term_behavior(self, wildcard_documents):
        result = await search(
            query="auth",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert "Exact Auth" in doc_names
        assert "Authentication Guide" not in doc_names
        assert "OAuth Guide" not in doc_names

    @pytest.mark.asyncio
    async def test_suffix_star_query_uses_prefix_matching(self, wildcard_documents):
        result = await search(
            query="auth*",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert "Exact Auth" in doc_names
        assert "Authentication Guide" in doc_names
        assert "OAuth Guide" not in doc_names

    @pytest.mark.asyncio
    async def test_explicit_fts_expression_uses_prefix_matching(self, wildcard_documents):
        result = await search(
            query="ignored",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
            fts_expression="auth*",
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert "Exact Auth" in doc_names
        assert "Authentication Guide" in doc_names
        assert "OAuth Guide" not in doc_names

    @pytest.mark.asyncio
    async def test_surrounded_star_query_uses_contains_matching(self, wildcard_documents):
        result = await search(
            query="*auth*",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert "Exact Auth" in doc_names
        assert "Authentication Guide" in doc_names
        assert "OAuth Guide" in doc_names

    @pytest.mark.asyncio
    async def test_explicit_regex_query_uses_regex_matching(self, wildcard_documents):
        result = await search(
            query="o?auth",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
            regex=True,
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert "Exact Auth" in doc_names
        assert "Authentication Guide" in doc_names
        assert "OAuth Guide" in doc_names

    @pytest.mark.asyncio
    async def test_explicit_regex_invalid_pattern_raises_value_error(self, wildcard_documents):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            await search(
                query="(",
                documents=wildcard_documents,
                top_k_docs=3,
                max_nodes_per_doc=5,
                search_mode="flat",
                regex=True,
            )

    @pytest.mark.asyncio
    async def test_unsupported_wildcard_shape_falls_back_to_plain_query(self, wildcard_documents):
        result = await search(
            query="au*th",
            documents=wildcard_documents,
            top_k_docs=3,
            max_nodes_per_doc=5,
            search_mode="flat",
        )
        doc_names = {doc["doc_name"] for doc in result["documents"]}
        assert doc_names == {"Exact Auth"}


class TestSearchSync:
    def test_sync_wrapper(self, sample_tree_structure):
        doc = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture and deployment guide.",
            structure=sample_tree_structure,
        )
        result = search_sync(
            query="backend",
            documents=[doc],
        )
        assert isinstance(result, dict)
        assert "flat_nodes" in result


class TestGrepFilterFallback:
    """Ensure GrepFilter works correctly when rg is unavailable."""

    def test_native_fallback_no_rg(self, sample_tree_structure):
        """GrepFilter falls back to native when rg is not installed."""
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=sample_tree_structure,
            metadata={"source_path": "/nonexistent/file.py"},
        )
        grep = GrepFilter([doc])
        with patch("treesearch.ripgrep.rg_available", return_value=False):
            scores = grep.score_nodes("FastAPI", doc.doc_id)
        # source_path doesn't exist, so it goes straight to native
        assert len(scores) > 0

    def test_native_fallback_no_source_path(self, sample_tree_structure):
        """GrepFilter uses native path when no source_path in metadata."""
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=sample_tree_structure,
        )
        grep = GrepFilter([doc])
        scores = grep.score_nodes("FastAPI", doc.doc_id)
        assert len(scores) > 0

    def test_lines_to_nodes_mapping(self):
        """Test that _lines_to_nodes correctly maps line numbers to node_ids."""
        structure = [
            {"title": "A", "node_id": "0", "text": "line1", "line_start": 1, "line_end": 5, "nodes": [
                {"title": "B", "node_id": "1", "text": "line2", "line_start": 2, "line_end": 3},
                {"title": "C", "node_id": "2", "text": "line3", "line_start": 4, "line_end": 5},
            ]},
        ]
        doc = Document(doc_id="t", doc_name="T", structure=structure)
        result = GrepFilter._lines_to_nodes(doc, [2, 3, 4])
        # Node "0" spans 1-5 so hits 2,3,4 → 3 hits
        # Node "1" spans 2-3 so hits 2,3 → 2 hits
        # Node "2" spans 4-5 so hits 4 → 1 hit
        assert "0" in result
        assert "1" in result
        assert "2" in result
        # Normalized: max is node "0" with 3 hits
        assert result["0"] == 1.0
        assert abs(result["1"] - 2 / 3) < 0.01
        assert abs(result["2"] - 1 / 3) < 0.01
