# -*- coding: utf-8 -*-
"""
Tests for TreeSearch.batch_search() and TreeSearch.abatch_search().

Covers:
  - Returns same number of results as queries
  - Results are in same order as queries
  - Results match individual search() calls
  - Empty query list returns empty list
  - Shares index (documents loaded once)
  - Async variant works correctly
  - Keyword arguments are forwarded to each search
"""
import os
import tempfile
import pytest

from treesearch import TreeSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DOC_A = """\
# Authentication

## Overview

The authentication system uses JWT tokens.

## Login Flow

1. User submits credentials.
2. Server validates and returns a JWT.
3. Client stores the token securely.
"""

_DOC_B = """\
# Database

## Schema

The database uses PostgreSQL with normalized tables.

## Migrations

Run migrations with alembic upgrade head.
"""

_DOC_C = """\
# Deployment

## Kubernetes

The application runs on Kubernetes with Helm charts.

## Scaling

Horizontal pod autoscaling is configured based on CPU.
"""


def _make_md_file(tmp_path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchSearch:
    def test_returns_list_same_length(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        ts = TreeSearch(db_path=None)
        ts.index(path_a, path_b)

        queries = ["JWT authentication", "database schema", "PostgreSQL"]
        results = ts.batch_search(queries)

        assert len(results) == len(queries)

    def test_order_preserved(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        ts = TreeSearch(db_path=None)
        ts.index(path_a, path_b)

        queries = ["JWT authentication", "database schema"]
        results = ts.batch_search(queries)

        # Each result should echo back its query
        for query, result in zip(queries, results):
            assert result["query"] == query

    def test_results_match_individual_search(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        ts = TreeSearch(db_path=None)
        ts.index(path_a, path_b)

        query = "JWT authentication login"
        individual = ts.search(query)
        batch = ts.batch_search([query])

        assert len(batch) == 1
        # flat_nodes count should be the same
        assert len(batch[0]["flat_nodes"]) == len(individual["flat_nodes"])

    def test_empty_queries_returns_empty(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        ts.index(path_a)

        results = ts.batch_search([])
        assert results == []

    def test_single_query(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        ts.index(path_a)

        results = ts.batch_search(["JWT token"])
        assert len(results) == 1
        assert "flat_nodes" in results[0]

    def test_kwargs_forwarded(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        ts.index(path_a)

        # max_nodes_per_doc=1 should limit results
        results = ts.batch_search(["JWT"], max_nodes_per_doc=1)
        for result in results:
            total_nodes = sum(len(d["nodes"]) for d in result["documents"])
            assert total_nodes <= 1

    def test_batch_vs_sequential_flat_node_content(self, tmp_path):
        """Each batch result should have the same top-1 node as individual search."""
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        path_c = _make_md_file(tmp_path, "deploy.md", _DOC_C)
        ts = TreeSearch(db_path=None)
        ts.index(path_a, path_b, path_c)

        queries = ["JWT", "PostgreSQL", "Kubernetes"]
        batch_results = ts.batch_search(queries, max_nodes_per_doc=3)
        for i, (q, batch_r) in enumerate(zip(queries, batch_results)):
            individual_r = ts.search(q, max_nodes_per_doc=3)
            # Top-1 node_id should match
            batch_top = batch_r["flat_nodes"][0]["node_id"] if batch_r["flat_nodes"] else None
            indiv_top = individual_r["flat_nodes"][0]["node_id"] if individual_r["flat_nodes"] else None
            assert batch_top == indiv_top, f"Query {q!r}: batch top={batch_top}, individual top={indiv_top}"

    def test_many_queries_all_return_results(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        ts = TreeSearch(db_path=None)
        ts.index(path_a, path_b)

        queries = ["JWT", "authentication", "login", "database", "schema", "migration",
                   "PostgreSQL", "alembic", "token", "credentials"]
        results = ts.batch_search(queries)

        assert len(results) == len(queries)
        for r in results:
            assert "flat_nodes" in r
            assert "query" in r

    def test_with_db_path(self, tmp_path):
        """batch_search works with on-disk db_path."""
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        db_path = str(tmp_path / "idx.db")
        ts = TreeSearch(db_path=db_path)
        ts.index(path_a)

        results = ts.batch_search(["JWT authentication", "login flow"])
        assert len(results) == 2

    def test_search_mode_forwarded(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        ts.index(path_a)

        results = ts.batch_search(["JWT"], search_mode="flat")
        assert results[0].get("mode") == "flat"


class TestBatchSearchAsync:
    async def test_abatch_search_returns_correct_count(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        await ts.aindex(path_a)

        queries = ["JWT", "login", "authentication"]
        results = await ts.abatch_search(queries)

        assert len(results) == 3

    async def test_abatch_search_order(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        path_b = _make_md_file(tmp_path, "db.md", _DOC_B)
        ts = TreeSearch(db_path=None)
        await ts.aindex(path_a, path_b)

        queries = ["authentication", "database schema", "PostgreSQL migrations"]
        results = await ts.abatch_search(queries)

        for q, r in zip(queries, results):
            assert r["query"] == q

    async def test_abatch_search_empty(self, tmp_path):
        path_a = _make_md_file(tmp_path, "auth.md", _DOC_A)
        ts = TreeSearch(db_path=None)
        await ts.aindex(path_a)

        results = await ts.abatch_search([])
        assert results == []
