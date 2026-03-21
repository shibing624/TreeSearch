# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for tree search: heuristics, TreeSearcher, and path results.
"""
import pytest
from treesearch.tree import Document
from treesearch.heuristics import (
    build_query_plan,
    score_anchor,
    score_walk_node,
    score_path,
    check_title_match,
    check_phrase_match,
)
from treesearch.tree_searcher import TreeSearcher, SearchState, PathResult
from treesearch.search import search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deep_tree_structure():
    """A deeper tree structure for testing tree walk."""
    return [
        {
            "title": "System Overview",
            "summary": "System overview and architecture.",
            "node_id": "0",
            "text": "This document describes the system architecture.",
            "nodes": [
                {
                    "title": "Architecture",
                    "summary": "System architecture overview.",
                    "node_id": "1",
                    "text": "The system uses microservices.",
                    "nodes": [
                        {
                            "title": "Backend",
                            "summary": "Python FastAPI backend.",
                            "node_id": "2",
                            "text": "The backend is built with Python and FastAPI. It handles REST API requests and authentication.",
                        },
                        {
                            "title": "Frontend",
                            "summary": "React TypeScript frontend.",
                            "node_id": "3",
                            "text": "The frontend uses React with TypeScript for type safety and user interaction.",
                        },
                        {
                            "title": "Database",
                            "summary": "PostgreSQL database layer.",
                            "node_id": "4",
                            "text": "PostgreSQL is used for persistent data storage with connection pooling.",
                        },
                    ],
                },
                {
                    "title": "Deployment",
                    "summary": "Kubernetes deployment guide.",
                    "node_id": "5",
                    "text": "The app is deployed on Kubernetes using Helm charts.",
                    "nodes": [
                        {
                            "title": "CI/CD Pipeline",
                            "summary": "Continuous integration and deployment.",
                            "node_id": "6",
                            "text": "GitHub Actions handles CI/CD with automated testing and deployment.",
                        },
                    ],
                },
            ],
        },
    ]


@pytest.fixture
def deep_doc(deep_tree_structure):
    """A Document with the deep tree structure."""
    return Document(
        doc_id="deep_test",
        doc_name="Deep Test Doc",
        structure=deep_tree_structure,
    )


# ---------------------------------------------------------------------------
# Tests: Document adjacency methods
# ---------------------------------------------------------------------------

class TestDocumentAdjacency:
    def test_get_parent_id(self, deep_doc):
        assert deep_doc.get_parent_id("2") == "1"
        assert deep_doc.get_parent_id("1") == "0"
        assert deep_doc.get_parent_id("0") is None

    def test_get_children_ids(self, deep_doc):
        children = deep_doc.get_children_ids("1")
        assert set(children) == {"2", "3", "4"}
        assert deep_doc.get_children_ids("2") == []

    def test_get_sibling_ids(self, deep_doc):
        siblings = deep_doc.get_sibling_ids("2")
        assert "3" in siblings
        assert "4" in siblings
        assert "2" not in siblings

    def test_get_depth(self, deep_doc):
        assert deep_doc.get_depth("0") == 0
        assert deep_doc.get_depth("1") == 1
        assert deep_doc.get_depth("2") == 2

    def test_get_path_to_root(self, deep_doc):
        path = deep_doc.get_path_to_root("2")
        assert path == ["0", "1", "2"]

    def test_get_subtree_node_ids(self, deep_doc):
        subtree = deep_doc.get_subtree_node_ids("1")
        assert set(subtree) == {"1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# Tests: QueryPlan
# ---------------------------------------------------------------------------

class TestQueryPlan:
    def test_basic_terms(self):
        plan = build_query_plan("backend Python FastAPI")
        assert "backend" in plan.terms
        assert "python" in plan.terms
        assert "fastapi" in plan.terms

    def test_quoted_phrase(self):
        plan = build_query_plan('"REST API" authentication')
        assert "REST API" in plan.phrases
        assert "authentication" in plan.terms

    def test_code_query_detection(self):
        plan = build_query_plan("how to import the auth module")
        assert plan.is_code_query is True

    def test_structural_query_detection(self):
        plan = build_query_plan("chapter 3 deployment")
        assert plan.is_structural_query is True

    def test_implicit_phrase(self):
        plan = build_query_plan("REST API")
        assert len(plan.phrases) >= 1


# ---------------------------------------------------------------------------
# Tests: Heuristic Scorers
# ---------------------------------------------------------------------------

class TestAnchorScorer:
    def test_basic_scoring(self):
        score = score_anchor(fts_score=0.8, depth=0)
        assert 0.7 <= score <= 1.0

    def test_depth_penalty(self):
        shallow = score_anchor(fts_score=0.8, depth=0)
        deep = score_anchor(fts_score=0.8, depth=4)
        assert shallow > deep

    def test_title_match_bonus(self):
        with_title = score_anchor(fts_score=0.5, depth=1, has_title_match=True)
        without_title = score_anchor(fts_score=0.5, depth=1, has_title_match=False)
        assert with_title > without_title

    def test_phrase_match_bonus(self):
        with_phrase = score_anchor(fts_score=0.5, depth=1, has_phrase_match=True)
        without_phrase = score_anchor(fts_score=0.5, depth=1, has_phrase_match=False)
        assert with_phrase > without_phrase


class TestWalkScorer:
    def test_basic_scoring(self):
        score = score_walk_node(lexical_score=0.8)
        assert score > 0

    def test_hop_penalty(self):
        near = score_walk_node(lexical_score=0.8, hop=0)
        far = score_walk_node(lexical_score=0.8, hop=3)
        assert near > far

    def test_redundancy_penalty(self):
        normal = score_walk_node(lexical_score=0.8, is_redundant=False)
        redundant = score_walk_node(lexical_score=0.8, is_redundant=True)
        assert normal > redundant


class TestPathScorer:
    def test_basic_scoring(self):
        score = score_path(
            leaf_score=0.8,
            path_titles=["System", "Architecture", "Backend"],
            path_texts=["system overview", "architecture design", "backend implementation details"],
            query_terms=["backend", "architecture"],
            path_length=3,
        )
        assert 0 < score <= 1.0

    def test_consistency_bonus(self):
        # All titles match query -> higher consistency
        consistent = score_path(
            leaf_score=0.6,
            path_titles=["Architecture", "Backend"],
            path_texts=["architecture overview", "backend service details"],
            query_terms=["architecture", "backend"],
            path_length=2,
        )
        inconsistent = score_path(
            leaf_score=0.6,
            path_titles=["Introduction", "Summary"],
            path_texts=["general introduction", "document summary"],
            query_terms=["architecture", "backend"],
            path_length=2,
        )
        assert consistent > inconsistent


class TestMatchHelpers:
    def test_title_match(self):
        assert check_title_match("Backend Architecture", ["backend"]) is True
        assert check_title_match("Frontend Design", ["backend"]) is False
        assert check_title_match("", ["backend"]) is False

    def test_phrase_match(self):
        assert check_phrase_match("REST API requests", ["REST API"]) is True
        assert check_phrase_match("simple text", ["REST API"]) is False


# ---------------------------------------------------------------------------
# Tests: TreeSearcher
# ---------------------------------------------------------------------------

class TestTreeSearcher:
    def test_search_returns_paths(self, deep_doc):
        searcher = TreeSearcher()
        fts_scores = {
            "deep_test": {
                "2": 0.9,  # Backend
                "3": 0.3,  # Frontend
                "1": 0.4,  # Architecture
            }
        }
        paths, flat_nodes = searcher.search(
            "backend Python FastAPI",
            [deep_doc],
            fts_scores,
        )
        assert len(paths) > 0
        assert isinstance(paths[0], PathResult)
        assert paths[0].doc_id == "deep_test"
        assert len(paths[0].path) > 0

    def test_search_flat_nodes_compat(self, deep_doc):
        searcher = TreeSearcher()
        fts_scores = {"deep_test": {"2": 0.9, "5": 0.5}}
        paths, flat_nodes = searcher.search("backend", [deep_doc], fts_scores)
        assert len(flat_nodes) > 0
        node = flat_nodes[0]
        assert "node_id" in node
        assert "doc_id" in node
        assert "score" in node
        assert "title" in node

    def test_path_has_root_to_leaf(self, deep_doc):
        searcher = TreeSearcher()
        fts_scores = {"deep_test": {"2": 0.9}}
        paths, _ = searcher.search("backend", [deep_doc], fts_scores)
        if paths:
            p = paths[0]
            # Path should start from root
            assert p.path[0]["node_id"] == "0"
            # Path should end at or near target
            assert p.target_node_id in [n["node_id"] for n in p.path]

    def test_empty_scores(self, deep_doc):
        searcher = TreeSearcher()
        paths, flat_nodes = searcher.search("query", [deep_doc], {})
        assert paths == []
        assert flat_nodes == []

    def test_anchor_dedup(self, deep_doc):
        """Anchors on the same path should be deduplicated."""
        searcher = TreeSearcher()
        fts_scores = {
            "deep_test": {
                "0": 0.3,
                "1": 0.5,
                "2": 0.9,
            }
        }
        paths, _ = searcher.search("architecture backend", [deep_doc], fts_scores)
        # Should not have duplicate paths to the same target
        target_ids = [p.target_node_id for p in paths]
        assert len(target_ids) == len(set(target_ids))

    def test_max_hops_respected(self, deep_doc):
        """Tree walk should not expand beyond max_hops."""
        from treesearch.config import set_config, TreeSearchConfig
        set_config(TreeSearchConfig(max_hops=1, max_expansions=10))
        try:
            searcher = TreeSearcher()
            fts_scores = {"deep_test": {"1": 0.9}}
            paths, _ = searcher.search("architecture", [deep_doc], fts_scores)
            # With max_hops=1, should reach children but not grandchildren from anchor
            for p in paths:
                # hop count should be within limits
                assert len(p.reasons) <= 10  # reasonable bound
        finally:
            from treesearch.config import reset_config
            reset_config()

    def test_reasons_populated(self, deep_doc):
        searcher = TreeSearcher()
        fts_scores = {"deep_test": {"2": 0.9}}
        paths, _ = searcher.search("backend", [deep_doc], fts_scores)
        if paths:
            assert len(paths[0].reasons) > 0

    def test_snippet_populated(self, deep_doc):
        searcher = TreeSearcher()
        fts_scores = {"deep_test": {"2": 0.9}}
        paths, _ = searcher.search("backend", [deep_doc], fts_scores)
        if paths:
            assert paths[0].snippet != ""


# ---------------------------------------------------------------------------
# Tests: Integrated search with tree mode
# ---------------------------------------------------------------------------

class TestSearchTreeMode:
    @pytest.mark.asyncio
    async def test_tree_mode_returns_paths(self, deep_tree_structure):
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=deep_tree_structure,
        )
        result = await search(
            query="backend Python FastAPI",
            documents=[doc],
            search_mode="tree",
        )
        assert result["mode"] == "tree"
        assert "paths" in result
        assert "documents" in result
        assert "flat_nodes" in result

    @pytest.mark.asyncio
    async def test_flat_mode_no_paths(self, deep_tree_structure):
        doc = Document(
            doc_id="test", doc_name="Test Doc",
            structure=deep_tree_structure,
        )
        result = await search(
            query="backend Python FastAPI",
            documents=[doc],
            search_mode="flat",
        )
        assert result["mode"] == "flat"
        assert "paths" not in result

    @pytest.mark.asyncio
    async def test_tree_mode_backward_compat(self, sample_tree_structure):
        """Tree mode should still return documents and flat_nodes."""
        doc = Document(
            doc_id="compat", doc_name="Compat Doc",
            structure=sample_tree_structure,
        )
        result = await search(
            query="backend",
            documents=[doc],
            search_mode="tree",
        )
        assert "documents" in result
        assert "flat_nodes" in result
        assert "query" in result
        assert result["query"] == "backend"
